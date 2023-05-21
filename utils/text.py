import re


def trim_motion_text_by_subtitle(row):
    """Remove leading text up to and including the subtitle"""
    try:
        split = row["text"].split(row["subtitle"], 1)
        return split[-1].strip()
    except Exception as e:
        print(e)
        return row["text"]


def trim_motion_text_by_leading_title(row):
    """Remove leading text up to and including the title"""
    if row["text"].startswith(row["title"]):
        text = row["text"].split(row["title"], 1)[-1].strip()
    else:
        text = row["text"]
    return text


def trim_whitespace(s):
    """Remove trailing and multiple whitespaces from a pandas series"""
    return s.replace("\s+", " ", regex=True).str.strip()


def trim_linebreaks(s):
    """Remove linebreaks and trailing whitespaces from a pandas series"""
    return s.replace("\n", " ", regex=True).replace("\r", " ", regex=True).str.strip()


def trim_motion_text_by_proposed_decision(row):
    """Remove leading text up to and including the proposed decision"""
    # .+?(?=(\. [A-ZÅÄÖ])) --> All up to first '.' followed by whitespace and
    # upper case letter. Not watertight by any means but a reasonable best effort.
    patterns = [
        r"Förslag till riksdagsbeslut .+?\. (?=([A-ZÅÄÖ]))",
        r"Riksdagen tillkännager för [A-Öa-ö]+ som sin mening .+?\. (?=([A-ZÅÄÖ]))",
        r"Riksdagen bemyndigar .+?\. (?=([A-ZÅÄÖ]))",
        r"Riksdagen beslutar om .+?\. (?=([A-ZÅÄÖ]))",
        r"Härmed hemställs att riksdagen .+?\. (?=([A-ZÅÄÖ]))",
        r"Med hänvisning till vad so(?:m|rn) anförts .+?\. (?=([A-ZÅÄÖ]))",
        r"Riksdagen ställer sig bakom det som anförs .+?\. (?=([A-ZÅÄÖ]))",
    ]
    split = re.split("|".join(patterns), row["text"])
    return split[-1].strip()


def trim_leadning_motivation(row):
    """Remove leading text up to and including the motivation header"""
    if re.match(r"^Motivering [A-ZÅÄÖ\d]", row["text"]):
        row["text"] = row["text"].split("Motivering", 1)[-1].strip()
    return row["text"]


def set_empty_when_leading_date(row):
    """Set string to empty if only place and date signature is left"""
    if re.match(r"^Stockholm den [\d]+ [a-z]+ \d{4}", row["text"]):
        row["text"] = ""
    return row["text"]


def delete_footer(row):
    """Remove footer"""
    return re.sub(r"(?<=\.) Stockholm den [\d]+ [a-z]+ \d{4} .+", "", row["text"])


def prep_text(df, has_title_cols=True):
    """Pipeline to preproces text column."""
    df["text"] = trim_linebreaks(df["text"])
    df["text"] = trim_whitespace(df["text"])
    if has_title_cols:
        df["text"] = df.apply(trim_motion_text_by_subtitle, axis=1)
        df["text"] = df.apply(trim_motion_text_by_leading_title, axis=1)
    df["text"] = df.apply(trim_motion_text_by_proposed_decision, axis=1)
    df["text"] = df.apply(trim_leadning_motivation, axis=1)
    df["text"] = df.apply(set_empty_when_leading_date, axis=1)
    df["text"] = df.apply(delete_footer, axis=1)
    return df["text"]
