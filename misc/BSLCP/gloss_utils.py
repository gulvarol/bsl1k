import re
import string
import functools
from typing import Set, Dict, Tuple, Union
from pathlib import Path
from collections import defaultdict

import tqdm
import numpy as np
from beartype import beartype
from typeguard import typechecked

import pympi
from intervaltree import Interval, IntervalTree


@typechecked
def parse_gloss(gloss: str, canonical_vocab: Set) -> Union[str, None]:
    """THe glossing system used for BSLCP is described in: `https://bslcorpusproject.org
    /wp-content/uploads/BSLCorpusAnnotationGuidelines_23October2014.pdf`

    The key parts for interpreting glosses are in Sec. 4.3. In the code below, relevant
    sections of the manual are quoted directly.

    NOTE: The ordering of the parsing is important. E.g. sign name prefixes must be
    parsed prior to finger spelling, because finger spelling can be nested inside a
    sign name, but not vice-versa (at least, I (Samuel) haven't seen any).
    """
    gloss = gloss.strip()

    # only apply a vocaulary filter when a non-empty vocabulary set is passed
    apply_vocab_filter = bool(canonical_vocab)

    # convert our existing vocabulary to upper cases to treat each word as a gloss
    upper_vocab = set([word.upper() for word in canonical_vocab])

    """
    Signs that correspond to multiple English words.
    ------------------------------------------------

    We do not use glosses which require multiple English words to describe a sign. The
    relevant section for these signs is:

    "All lexical signs are annotated using an identifying gloss (ID gloss) from BSL
    SignBank if one exists (if one does not, see 6.4). An ID gloss is an English gloss
    (always in upper case, e.g., SISTER) that is consistently used with a unique sign
    (or ‘lemma’) to represent the sign whether in citation form or any phonological or
    morphological variant. If a sign needs more than one distinct English word to gloss
    it, hyphens are placed between the words (spaces are not used), e.g. PULL-APART not
    PULL APART OR PULLAPART. The ID gloss for each sign (in citation form) can be found
    in the BSL SignBank and is usually the same as one of the keywords associated with
    the sign. It is important that BSL SignBank is consulted to ensure that the right
    ID gloss is used at all times."
    """
    if "-" in gloss:
        return None

    """
    Glosses with multiple possibile interpretations
    ---
    When a slash is present in a sign, we consider both options as candidate glosses.
    The relevant section in the manual is:

    "If two possibilities exist for a single token and it is difficult to decide between
    the two, then both ID glosses are entered with a forward slash between each token
    (e.g., LOOK/THINK). More than two ID 12 glosses are entered, each separated by a
    forward slash, if more than two possibilities exist (e.g., LOOK/THINK/SEE). This
    convention is also used to demonstrate ambiguity between sign types - for example,
    when one is uncertain if a given token is a lexical sign or a sequence of
    constructed action (e.g., TEA/G:CA:DRINK-TEA)."
    """
    gloss_candidates = gloss.split("/")

    for gloss in gloss_candidates:

        """
        Compound words
        ---
        We do not make use of compund words that are stored as caret-separated tokens.
        The relevant section in the manual is:

        Most compounds are found with distinct ID-glosses in BSL SignBank, e.g., the
        compound combining MOTHER and FATHER is a standard compound PARENTS. If a pairing
        of signs cannot be found in BSL SignBank as a compound, the two signs are
        tentatively annotated as one sign with two ID-glosses but separated with a caret
        (e.g. FS:G-GRAPHIC^ART for 'graphic designer'). Each of these tokens will
        be returned to later and will either be given its own ID-gloss if it is found to
        be in widespread use or treated as two signs to be annotated and glossed
        separately. In some cases, these tokens may be best regarded as a collocation.
        (A collocation could be a potential compound if the overall meaning is not
        predictable from the two signs that are paired. A collocation is unlikely to
        be a compound if it is possible to insert another sign between the two signs.)
        """
        if "^" in gloss:
            continue

        """
        Space delimited words
        ---

        Since it is not clear how to deal with glosses containing spaces, we discard
        them.
        """
        if " " in gloss:
            continue

        """Gesture signs
        ---
        For most gesture signs, we do not apply any special processing (we simply
        remove the gesture annotation (the 'G:' prefix).  Two exceptions to this rule
        are:
        1. "gesture constructed actions" which are prefixed by G:CA and which are removed
        (as non-lexical).
        2. Non-manual gestures (prefixed by G(NMS):) and are removed.

        The relevant section for gestures in the manual is:

        "4.8.3 Gesture
        • All gesture annotations begin with ‘G:’ followed by a brief description of its
        meaning (not form – i.e. G:HOW-STUPID-OF-ME and not G:HIT-PALM-ON-FOREHEAD).
        • The gesture with upturned hands (also known as the ‘palm-up gesture’) is
        annotated as G:WELL. This is the second most frequent token in the BSL Corpus to
        date.
        • Some emblems are lexicalised and glossed as lexical signs without the gesture
        prefix (e.g., GOOD) although this is not always the case (e.g., G:FUCK-OFF).
        Whether with the gesture prefix or not, emblems have been added and are being
        added to BSL SignBank. As with G:WELL, the form associated with G:FUCK-OFF is
        consistently recognised using the same gloss (also provided in BSL SignBank).
        This is in recognition of the fact that they appear to have consistent
        form/meaning mappings even though their lexical status is unclear.
        • Tokens of constructed action are also recognised as instances of gestural
        activity. Such instances are marked with the prefix ‘G:CA:’. As with classifier
        signs and other types of gesture, this prefix is followed by a brief description
        of the token’s meaning (e.g., G:CA:HOLD-HANDS-UP-IN-FRIGHT). CA tokens are not
        lexicalised and thus are not included in BSL SignBank.
        • Sequences of constructed action can appear to be difficult to separate from the
        category of handling classifier signs. Within the BSL Corpus, all instances where
        the handshape mimics the actual handshape used to carry out the activity in real
        life are annotated as constructed action sequences first by default, following
        Cormier et al. (2012). Additional evidence is required to label the token as a
        handling classifier sign. This may include instances of modification that appear
        to be typical of lexical verbs (e.g., aspectual modification)."

        A note about the G:CA prefix:
        "Tokens of constructed action are also recognised as instances of gestural
        activity. Such instances are marked with the prefix ‘G:CA:’. As with classifier
        signs and other types of gesture, this prefix is followed by a brief description
        of the token’s meaning (e.g., G:CA:HOLD-HANDS-UP-IN-FRIGHT). CA tokens are not
        lexicalised and thus are not included in BSL SignBank.

        G(NMS) is not described in the manual, but presumably indicates "non-manual signs"
        We do not include these.
        "
        """
        # ignore gestures of constructed action and non-manual signs
        prefixes = ["G:CA:", "G(NMS):"]
        if any([gloss.startswith(prefix) for prefix in prefixes]):
            continue

        prefix = "G:"
        if gloss.startswith(prefix):
            gloss = gloss[len(prefix):]

        """Constructed actions
        ---
        Constructed actions are "a stretch of discourse that represents one role or
        combination of roles depicting actions, utterances, thought, attitudes and/or
        feelings of one or more referents)" (source:
        https://discovery.ucl.ac.uk/id/eprint/1447790/1/Cormier_rethinking_constructed_
        action.pdf)

        They are prefixed by "CA:". As noted above, CA tokens are not lexicalized (and
        are not included in SignBank), so we also do not make use of them.
        """
        prefix = "CA:"
        if gloss.startswith(prefix):
            continue

        """
        Uncertain signs
        ---------------
        We do not make use of uncertain signs. The relevant section in the manual is:

        "If the identity of a sign is uncertain, but a possibility exists, then the ID
        gloss is entered prefixed by a ‘?’ (e.g., ?HOME). This may be used because the
        token looks like it could be a phonological variant of this lemma or it may be
        a separate (new) lemma.

        Occasionally, a gloss ends with a question mark (e.g. HOME02?). This is not
        documented in the annotation guidelines, but we also ignore these.

        Even more rarely, the question mark can appear in the middle (e.g.
        'POISON?(POISON)'): these are also ignored.
        """
        token = "?"
        if token in gloss:
            continue

        """
        Sign names
        ---

        We make use of sign names when the name also corresponds to a lexical sign.
        These take the form: SN:NAME(LEXICAL_SIGN) (here we would keep LEXICAL_SIGN).
        The relevant section in the manual is:

        "4.5 Sign Names

        NOTE: It is likely that the annotation of sign names has not been fully
        consistent with these guidelines in the current release. For future releases we
        will aim to make these more consistent.
        • Signs name are entered with the prefix SN: followed by the proper name.
        The sign name for a person called Peter would be written as follows: SN:PETER,
        unless the sign name is identical in form to a lexical sign (see below).
        • If the sign name is identical in form to a lexical sign, the ID gloss for the
        relevant sign may be identified after the name in brackets: e.g.,
        SN:MISS-JENKINS(HAIR-BUN), SN:WEMBLEY(STADIUM) or SN:OSAMA-BIN-LADEN(BEARD).
        • If the sign name uses a lexical sign that is in BSL SignBank but the annotator
        is unable to determine the name of the referent in this instance then the gloss
        UNKNOWN is used (e.g., SN:UNKNOWN(WOLF)).
        • If the sign name is based on fingerspelling, the form is entered in brackets
        after the name and follows the conventions for fingerspelling as outlined below:
        e.g., SN:PETER(FS:PETER(PR)), (SN:PETER(FS:P-PETER), or
        SN:ALEX-FERGUSON(FS:A-ALEX^FS:F-FERGUSON)).
        • If the sign name represents a sequence of both fingerspelling and a lexical
        sign, the whole sequence is entered as one sign name. The fingerspelled element
        and lexical element are included in brackets separated by a caret (e.g.,
        SN:JOHN-KING(FS:J-JOHN^KING)).
        • It can be difficult to determine when a fingerspelled sequence is in fact a
        sign name. Generally, we assume that fingerspelled sequences that use the initial
        letter of the name or fingerspelled sequences that are reduced so that they appear
        like lexical signs are sign names that have some degree of conventionalisation.
        Therefore, fully fingerspelled sequences (e.g., FS:BARRY where each letter is
        articulated clearly) are typically entered as fingerspelled sequences and
        not sign names (i.e., they do not have the prefix SN attached to them).
        • Sign names are often for people but may be for e.g. places, organisations, etc.
        Some signs names however have been judged to be institutionalised and consistently
        used (some more than others) across the British Deaf community and thus are
        included in BSL SignBank as lexical signs (e.g., LONDON, BRISTOL, SEE-HEAR,
        DEAFINTELY-THEATRE).
        • If the annotator cannot determine what the signed sequence represents, the
        glosses INDECIPHERABLE or unknown may be used (e.g.,
        SN:INDECIPHERABLE(FS:INDECIPHERABLE(H)), SN:UNKNOWN(UNKNOWN))."
        """
        # ignore indicpherable glosses
        if gloss.startswith("SN:INDECIPHERABLE"):
            continue

        regexp = r"^SN:([A-Z]+)\((?P<lexical_gloss>[A-Z0-9:\(\)]+)\)$"
        matches = re.match(regexp, gloss)
        if matches:
            gloss = matches.group("lexical_gloss")
            # ignore unknown glosses that have been parsed from sign names
            if gloss == "UNKNOWN":
                continue

        # keep all other standard Sign Names
        if gloss.startswith("SN:"):
            gloss = gloss[len("SN:"):]

        """
        Fingerspelling
        ---
        We keep most fingerspellings, but remove those that are marked as "INDECIPHERABLE"
        If a word contains a meaning that is inferred e.g. FS:INFERRED_WORD(BIT_OF_WORD),
        we simply keep the INFERRED_WORD. The relevant manual section is below:

        "4.8.4 Fingerspelling

        Fingerspelled forms in BSL represent a sequence of hand configurations that have
        a one-to-one correspondence with the letters of the English alphabet.
        Fingerspelled forms often violate phonological constraints associated
        with core native signs and are said to belong to what is known as the
        ‘non-native lexicon’ (Brentari & Padden, 2001). Below are the conventions used
        for fingerspelled sequences whilst annotating the BSL Corpus.
        • Fingerspelling is annotated with the fingerspelled word prefixed with FS for
        ‘fingerspelling’ followed by a colon and then the word spelled, e.g., FS:WORD.
        • If not all the letters of a word are spelled, but it is clear what word the
        signer is attempting to fingerspell, the full spelling of the intended word is
        entered (not the misspelling – e.g. FS:WORD(WRD) and not FS:WRD).
        • If not all the letters of a word are spelled, and it is not clear what
        word the signer is attempting to fingerspell, the gloss INDECIPHERABLE is used
        followed by the actual letters produced in brackets (e.g.,
        FS:INDECIPHERABLE(GTH)) or FS:B-INDECIPHERABLE).
        • If the fingerspelling is for multiple words, a new annotation per word is
        begun even if it is one continuous act of fingerspelling (e.g.,
        FS:MISS FS:JENKINS and not FS:MISSJENKINS).
        • If the form is a single fingerspelled letter (or single fingerspelled letter
        repeated), the letter and the word it stands for are included in the annotation.
        In this case, the fingerspelled letter precedes the word it represents
        (e.g., FS:F-FORTUNE, FS:C-CONTRIBUTION). These sequences are also known as single
        manual letter signs (SMLS signs). Even if the single manual letter is repeated,
        only one English letter is included in the annotation (e.g., FS:F-FORTUNE not
        FS:FF-FORTUNE).
        • Some SMLS or fingerspelled sequences are actually lexical signs in their own
        right. These include MOTHER, YEAR, YELLOW, DAUGHTER and CLUB. Although they
        are based on fingerspelling, these signs do not have the prefix ‘FS:’ because
        we do not use this prefix when the sign in question is a fully lexical sign.
        These signs are in BSL SignBank.
        • The distinction between fingerspelled forms and lexicalised fingerspelled signs
        is often difficult to maintain given that many fingerspelled forms can appear
        partly nativised (i.e., may be in the process of becoming a fully lexical sign).
        For example, the sign SATURDAY3 is a fingerspelled loan (based on SA-T) that is
        considered partly nativised for it does not follow constraints imposed on fully
        lexical signs - e.g., it violates the selected fingers constraint
        (Brentari, 1998). To categorise these tokens in a principled way, we use
        guidelines based on Cormier et al. (2008). Fingerspelled loans (with remnants of
        2 or more letters) are accepted as lexical signs if there is evidence of
        phonological restructuring to make them more native-like. Additionally,
        independent agreement from native signers can also be sought to be certain that
        this form is consistently used for that meaning (e.g., SATURDAY3 was accorded
        lexical status following these criteria)."
        """
        # ignore indicpherable glosses
        if gloss.startswith("FS:INDECIPHERABLE"):
            continue

        # keep inferered glosses
        regexp = r"^FS:(?P<inferred_word>[A-Z0-9]+)\([A-Z]+\)$"
        matches = re.match(regexp, gloss)
        if matches:
            gloss = matches.group("inferred_word")

        # keep all other standard finger spellings
        if gloss.startswith("FS:"):
            gloss = gloss[len("FS:"):]

        """
        Depicting signs
        ---
        These don't appear to be described in the manual. They use a DSS or DS prefix,
        followed by the meaning in parentheses. E.g. DSS(FLAT). Sometimes they also
        include a colon (e.g. DSS:(SMALL_OPEN)). We do not include them.


        NOTE: There is a description of DS/DSS signs under the section entitled
        "partly/non-lexical material" in
        https://bslcorpusproject.org/wp-content/uploads/Bibibi_Digging_Handout.pdf
        which contains the following statement:

        “ For example, the annotation for a depicting sign indicates its category (DS),
        and the additional information conveys a rough approximation to meaning:
        DS(vehicle-goes-down-street).”

        Thus, it appears to mean that the sign depicts what is happening with gestures.
        """
        # keep depicting signs
        regexp = r"^DS[S]*\(*[A-Z0-9_]*\)*[:]*\(*[A-Z0-9_]+\)*$"
        matches = re.match(regexp, gloss)
        if matches:
            continue

        """
        Fragment buoys
        ---
        These are typically denoted by `FBUOY` or `Fbuoy`. They are discarded.

        Notes from manual:
        "With a fragment buoy, the non-dominant hand is held from a preceding sign, it is
        intended, and it carries some meaning. Typically this meaning is conveyed by the
        signer pointing to or looking at or directing attention to the fragment buoy in
        some way. This differs from perseveration whereby the nondominant hand is held
        from a preceding sign but may or may not be intended to carry any meaning. 
        This is annotated as FBUOY. Note: There are some known mistakes in annotation
        of fragment buoys in the first release; some of them are actually LBUOYs.
        If the signer points to a fragment buoy with the dominant hand, the point is
        annotated as PT:FBUOY."
        """
        tokens = ["FBUOY", "Fbuoy"]
        if any([token in gloss for token in tokens]):
            continue

        """
        List buoys
        ---
        These are typically denoted by `LBUOY` or `Lbuoy`. They are discarded.

        Notes from the manual on List buoys:
        "• When producing a list buoy, a certain number of fingers are held stretched
        out on the non-dominant hand, each one referring to an entity or idea, that are
        all somehow related, often sequentially. For example, an index finger can be
        held up to indicate the first of a series of items. The list buoy is annotated
        as LBUOY in each instance.
        • If the signer points to a list buoy with the dominant hand, the point is
        annotated as PT:LBUOY."
        """
        tokens = ["LBUOY", "Lbuoy"]
        if any([token in gloss for token in tokens]):
            continue

        """
        Classifier/depicting signs
        ---
        These are slight variants on the depicting signs described above. They are
        not kept.

        Notes from: "Digging into Signs: Towards a gloss annotation standard for sign
        language corpora":

        "Basically, classifier/depicting signs are annotated with one of four elements
        for movement (MOVE, PIVOT, AT, BE), followed by classifier handshape. Additional
        prefixes are added for the type of
        depicting sign: whole entity (DSEW), part entity (DSEP), or handling (DSH).
        Examples: BSL: DSEW(2)-MOVE DSEP(1)-PIVOT DSEW(2)-AT
        """
        prefixes = ["DSEW", "DSH", "DSEP"]
        if any([gloss.startswith(prefix) for prefix in prefixes]):
            continue

        """
        Lexical variants
        ---
        We combine lexical variants into a single gloss. The relevant section for these
        is:

        "Lexical variants have the same or similar/related meanings but (unlike
        phonological
        variants) they generally differ in two parameters or more from each other.
        Lexical variants are distinguished using a numeral tag (e.g., BROWN, BROWN2
        and BROWN3). Note that the first lexical variant is not indicated with a number
        (e.g., BROWN not BROWN1). (This is simply to aid in the speed of annotation,
        eliminating the need to find/replace all tokens of e.g. BROWN with BROWN1 in
        ELAN and the same change made in BSL SignBank, when BROWN2 is encountered.)"
        """
        regexp = r"^(?P<base_gloss>[A-Z]+)[0-9]+$"
        matches = re.match(regexp, gloss)
        if matches:
            gloss = matches.group("base_gloss")

        """
        Pointing signs
        ---
        Pointing signs are used to reference things.  We do not include them.

        NOTE: They are described in section 4.8.1 of the manual. The section is v. long,
        so it is not included here.
        """
        # ignore pointing glosses
        if gloss.startswith("PT:"):
            continue

        """
        Classifier signs/locations
        ---
        These are glosses prefixed by "CLL:", "CLH:", "CLS. We do not include them.

        Described in the manual:
        CLL: "Depicts the location of entities, often by a short movement at a location
              or a hold."
        CLM: "Depicts the movement of entities"
        CLH: "Depicts the handling of an object"
        CLSS: "Depicts the size and shape of entities, most often with a tracing movement
              but also sometimes with a hold"
        """
        prefixes = ["CLL", "CLM", "CLH", "CLSS"]
        if any([gloss.startswith(prefix) for prefix in prefixes]):
            continue

        """
        Malformed or unrecognised glosses
        ---

        1. If a gloss ends in a colon, we ignore it.
        2. If a gloss does not have a matching number of left and right parentheses, we
           ignore it.
        """
        if gloss.endswith(":"):
            continue

        if gloss.count("(") != gloss.count(")"):
            continue

        """
        Undocumented
        ---

        1.Sometimes the gloss ends with a lowercase alphabetical character. It seems
        reasonable to assume that these are lexical variants (although it is hard to
        find this in the documentation.) We treat them as lexical variants, and check
        that there is only a single such letter.

        2. If a gloss contains a parentheses, but not additional suffix or prefix, we
        discard it, because it is not clear how it should be interpreted.

        3. If a gloss contains a multiplier of the form GLOSSxN, where N is a number, we
        discard it, because it is not clear how it should be interpreted.

        4. If a gloss contains an equals sign (e.g. 'TEN05=1') we discard it,
        because it is not clear how it should be interpreted.

        5. If a gloss contains an asterisk sign (e.g. 'SIXTEEN(A*z') we discard it,
        because it is not clear how it should be interpreted.

        6. If a gloss is prefixed with "F:" (e.g. F:FOR), we discard it,
        because it is not clear how it should be interpreted.

        7. If a gloss is prefixed with "M:" (e.g. M:WHY), we discard it,
        because it is not clear how it should be interpreted.

        """
        if any([gloss.endswith(x) for x in string.ascii_lowercase]):
            gloss = gloss[:-1]
            # If, after removing a loweer case suffix, not all remaining characters are
            # uppercase, we discard the gloss.
            if gloss != gloss.upper():
                continue

        # discard glosses with parentheses but no suffix or prefix
        regexp = r"^[A-Z]+\([a-zA-Z0-9]+\)$"
        matches = re.match(regexp, gloss)
        if matches:
            continue

        # discard glosses with multipliers anywhere in the sequence
        regexp = r"[A-Z]*x[0-9]+[A-Z]*"
        matches = re.search(regexp, gloss)
        if matches:
            continue

        # discard glosses containing an equals signs
        if "=" in gloss:
            continue

        # discard glosses containing an asterisk
        if "*" in gloss:
            continue

        # if gloss starts with unknown prefix, skip
        if gloss.startswith("F:") or gloss.startswith("M:"):
            continue

        # if no member of the supplied vocabulary is present in any part of the gloss,
        # we reject the word
        if apply_vocab_filter and not any([word in gloss for word in upper_vocab]):
            continue

        # full words can be directly checked against the vocabulary
        regexp = r"^(?P<full_word>[A-Z']+)$"
        matches = re.match(regexp, gloss)
        if matches:
            gloss = matches.group("full_word")
            if gloss in upper_vocab or not apply_vocab_filter:
                return gloss
            else:
                continue
    return None


@functools.lru_cache(maxsize=64, typed=False)
def filter_glosses(gloss_data: Dict, canonical_vocab: Tuple) -> Dict:
    """We perform two stages of filtering on the glosses:
    1. Filter to the given vocabulary
    2. Remove duplicates that come from annotating the LH and RH glosses separately.
    """
    filtered = {}
    canonical_vocab = set(canonical_vocab)
    for gloss, data in tqdm.tqdm(gloss_data.items()):
        parsed_gloss = parse_gloss(gloss, canonical_vocab)
        if parsed_gloss:
            if parsed_gloss in filtered:
                for key, val in data.items():
                    filtered[parsed_gloss][key].extend(val)
            else:
                filtered[parsed_gloss] = data
    return filtered


@beartype
def filter_left_right_duplicates(data: dict, iou_thr: float = 0.9) -> dict:
    """Filter LH and RH duplicate glosses

    Args:
        data: gloss annotations
        iou_thr: the intersection-over-union threshold used to supress duplicated

    Returns:
        filtered gloss data

    NOTES: One challenge with the data is that often glosses that use two hands are
    annotated twice, once as `LH-IDgloss` and once as `RH-IDgloss`. These "duplicate"
    glosses have very similar, but not identical start and end times (so we cannot simply
    use hashing to filter them out). Instead, we use an interval tree which allows us to
    efficiently query the overlap between start and end times.

    The algorithm is:
    - for each gloss (which will either be of type `LH-IDgloss` or `RH-IDgloss`, check
      if it's "pair" (i.e. the pair of LH-IDgloss would be RH-IDgloss, and vice versa)
      exists in the interval tree with "high" overlap.  If so, exclude this gloss.
      Otherwise add it to the tree.
    """
    lookup_pairs = {
        "LH-IDgloss": "RH-IDgloss",
        "RH-IDgloss": "LH-IDgloss",
    }

    msg = "All gloss data lists of values should have the same length"
    assert len({len(val) for val in data.values()}) == 1, msg

    keep = np.zeros(len(data["start"]), dtype=np.bool)
    gloss_interval_trees = {}
    zipped = zip(data["start"], data["end"], data["media"], data["tier"])
    for ii, (begin, end, media, tier) in enumerate(zipped):

        # Media is often supplied as a list of videos, so we merge them into a single
        # hashable string to ensure we only suppress annotations within the same video
        media_key = " ".join(media)
        if media_key not in gloss_interval_trees:
            gloss_interval_trees[media_key] = {
                key: IntervalTree() for key in lookup_pairs}

        paired_gloss = lookup_pairs[tier]
        # check if the current gloss has already been annotated for the other hand
        overlaps = gloss_interval_trees[media_key][paired_gloss].overlap(begin, end)
        if overlaps:
            overlap_exceeded = False
            for interval in overlaps:
                intersection = min(end, interval.end) - max(begin, interval.begin)
                union = max(end, interval.end) - min(begin, interval.begin)
                iou = intersection / union
                if iou >= iou_thr:
                    overlap_exceeded = True
                    break
            keep[ii] = not overlap_exceeded
        else:
            interval = Interval(begin=begin, end=end, data=None)
            gloss_interval_trees[media_key][tier].add(interval)
            keep[ii] = 1
    return {key: np.array(val)[keep].tolist() for key, val in data.items()}


@functools.lru_cache(maxsize=64, typed=False)
def parse_glosses(
        anno_dir: Path,
        target_tiers: Tuple[str, str],
) -> Dict:
    """Parse ELAN EAF files into a python dictionary.

    Args:
        anno_dir: the root directory containing all eaf files to be parsed.
        target_tiers: the names of the tiers (e.g. "RH-IDgloss") to be parsed.

    Returns:
        the parsed data
    """
    parsed = defaultdict(lambda: {"start": [], "end": [], "media": [], "tier": []})
    anno_paths = list(anno_dir.glob("**/*.eaf"))
    print(f"Found {len(anno_paths)} in {anno_dir}")
    seen = set()
    count = 0
    for anno_path in tqdm.tqdm(anno_paths):
        eafob = pympi.Elan.Eaf(str(anno_path))

        # Katrin's timing fix
        media_paths = [x["RELATIVE_MEDIA_URL"] for x in eafob.media_descriptors]
        stems = [Path(x.replace("-comp", "").replace(".compressed", "")).stem
                 for x in media_paths]
        assert len(stems) in {1, 2}, "Expected 1 or 2 media paths"
        if len(stems) == 2:
            offset_id = 0 if len(stems[0]) <= len(stems[1]) else 1
        else:
            offset_id = 0
        time_origin = int(eafob.media_descriptors[offset_id].get("TIME_ORIGIN", 0))

        for tier in target_tiers:
            for annotation in eafob.get_annotation_data_for_tier(tier):
                start, end = [(x + time_origin) / 1000 for x in annotation[:2]]
                keyword = annotation[2]
                media_paths = [x["RELATIVE_MEDIA_URL"] for x in eafob.media_descriptors]
                key = (start, end, "-".join(sorted(media_paths)), tier)
                if key in seen:
                    continue
                seen.add(key)
                parsed[keyword]["start"].append(start)
                parsed[keyword]["end"].append(end)
                parsed[keyword]["media"].append(media_paths)
                parsed[keyword]["tier"].append(tier)
                count += 1
    return dict(parsed)
