"""Turn a place as the user wrote it into an OpenWeather city query (no network calls).

OpenWeather's ``q`` parameter wants ``City``, ``City,CC`` (country code), or ``City,ST,US`` for a
US state. People write "amarillo texas", "Amarillo, TX", or "Paris, France", and "Amarillo, TX"
returns a 404 as-is (TX is read as a country). ``normalize_place`` converts those forms.

Ambiguity: OpenWeather's geocoder returns several matches for nearly every name (Denver, Paris,
London, Austin, and Boston included), and the weather endpoint silently picks its top match. Its
top match is the well-known city for those names, so a bare name is used as-is, except for the
curated ``AMBIGUOUS_CITIES`` whose name is genuinely split between places people mean (Portland,
Springfield, ...). For those, a bare name gets a clarifying question listing the options.
"""

from __future__ import annotations

import re
from typing import Any

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
US_STATE_CODES = set(US_STATES.values())

COUNTRIES = {
    "usa": "US", "us": "US", "united states": "US", "america": "US", "canada": "CA", "mexico": "MX",
    "france": "FR", "england": "GB", "scotland": "GB", "wales": "GB", "uk": "GB", "united kingdom": "GB",
    "britain": "GB", "great britain": "GB", "ireland": "IE", "germany": "DE", "italy": "IT", "spain": "ES",
    "portugal": "PT", "netherlands": "NL", "belgium": "BE", "switzerland": "CH", "austria": "AT",
    "sweden": "SE", "norway": "NO", "denmark": "DK", "finland": "FI", "poland": "PL", "greece": "GR",
    "turkey": "TR", "russia": "RU", "japan": "JP", "china": "CN", "south korea": "KR", "korea": "KR",
    "india": "IN", "australia": "AU", "new zealand": "NZ", "brazil": "BR", "argentina": "AR", "chile": "CL",
    "peru": "PE", "colombia": "CO", "uruguay": "UY", "jamaica": "JM", "costa rica": "CR", "egypt": "EG",
    "south africa": "ZA", "kenya": "KE", "israel": "IL", "uae": "AE", "united arab emirates": "AE",
}

# Names people commonly mean for more than one place. Option: (label, OpenWeather query).
AMBIGUOUS_CITIES: dict[str, list[tuple[str, str]]] = {
    "portland": [("Portland, Oregon", "Portland,OR,US"), ("Portland, Maine", "Portland,ME,US")],
    "springfield": [("Springfield, Illinois", "Springfield,IL,US"), ("Springfield, Missouri", "Springfield,MO,US"),
                    ("Springfield, Massachusetts", "Springfield,MA,US")],
    "columbus": [("Columbus, Ohio", "Columbus,OH,US"), ("Columbus, Georgia", "Columbus,GA,US")],
    "kansas city": [("Kansas City, Missouri", "Kansas City,MO,US"), ("Kansas City, Kansas", "Kansas City,KS,US")],
    "arlington": [("Arlington, Texas", "Arlington,TX,US"), ("Arlington, Virginia", "Arlington,VA,US")],
    "aurora": [("Aurora, Colorado", "Aurora,CO,US"), ("Aurora, Illinois", "Aurora,IL,US")],
    "rochester": [("Rochester, New York", "Rochester,NY,US"), ("Rochester, Minnesota", "Rochester,MN,US")],
    "salem": [("Salem, Oregon", "Salem,OR,US"), ("Salem, Massachusetts", "Salem,MA,US")],
    "charleston": [("Charleston, South Carolina", "Charleston,SC,US"),
                   ("Charleston, West Virginia", "Charleston,WV,US")],
    "wilmington": [("Wilmington, Delaware", "Wilmington,DE,US"), ("Wilmington, North Carolina", "Wilmington,NC,US")],
    "jackson": [("Jackson, Mississippi", "Jackson,MS,US"), ("Jackson, Tennessee", "Jackson,TN,US")],
    "lafayette": [("Lafayette, Louisiana", "Lafayette,LA,US"), ("Lafayette, Indiana", "Lafayette,IN,US")],
    "greenville": [("Greenville, South Carolina", "Greenville,SC,US"),
                   ("Greenville, North Carolina", "Greenville,NC,US")],
    "richmond": [("Richmond, Virginia", "Richmond,VA,US"), ("Richmond, California", "Richmond,CA,US")],
    "albany": [("Albany, New York", "Albany,NY,US"), ("Albany, Georgia", "Albany,GA,US")],
    "birmingham": [("Birmingham, Alabama", "Birmingham,AL,US"), ("Birmingham, England", "Birmingham,GB")],
    "cambridge": [("Cambridge, Massachusetts", "Cambridge,MA,US"), ("Cambridge, England", "Cambridge,GB")],
    "manchester": [("Manchester, England", "Manchester,GB"), ("Manchester, New Hampshire", "Manchester,NH,US")],
    "vancouver": [("Vancouver, British Columbia", "Vancouver,CA"), ("Vancouver, Washington", "Vancouver,WA,US")],
    "athens": [("Athens, Greece", "Athens,GR"), ("Athens, Georgia", "Athens,GA,US")],
    "san jose": [("San Jose, California", "San Jose,CA,US"), ("San Jose, Costa Rica", "San Jose,CR")],
    "durham": [("Durham, North Carolina", "Durham,NC,US"), ("Durham, England", "Durham,GB")],
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" \t?!.;:'\"")).strip()


def _title(name: str) -> str:
    return " ".join(w if w.isupper() and len(w) > 1 else w[:1].upper() + w[1:] for w in name.split())


def _qualifier_code(qualifier: str, *, allow_abbreviation: bool) -> tuple[str, str] | None:
    """('state', 'TX') / ('country', 'FR') for a qualifier such as 'texas', 'TX', 'France'."""
    q = qualifier.lower().strip(" .")
    if q in US_STATES:
        return "state", US_STATES[q]
    if q in COUNTRIES:
        return "country", COUNTRIES[q]
    if allow_abbreviation and len(q) == 2 and q.upper() in US_STATE_CODES:
        return "state", q.upper()  # this app is US-first: 'CA' means California, not Canada
    if allow_abbreviation and q in ("d.c", "dc"):
        return "state", "DC"
    return None


def normalize_place(text: str) -> dict[str, Any]:
    """``{"name", "query", "label", "qualified", "ambiguous": [(label, query), ...]}`` for a place.

    ``query`` is what to pass to the weather tool. ``ambiguous`` is non-empty only for a bare name
    in ``AMBIGUOUS_CITIES``.
    """
    text = _clean(text)
    name, kind_code = text, None
    if "," in text:
        head, _, tail = text.partition(",")
        kind_code = _qualifier_code(tail.split(",")[0], allow_abbreviation=True)
        if kind_code or tail.strip():
            name = head.strip()
    elif text.lower() not in US_STATES and text.lower() not in COUNTRIES:  # "west virginia" is not West, VA
        words = text.split()
        # Trailing state or country name ("amarillo texas", "paris france", "new york new york").
        for size in (3, 2, 1):
            if len(words) > size:
                found = _qualifier_code(" ".join(words[-size:]), allow_abbreviation=size == 1)
                if found:
                    name, kind_code = " ".join(words[:-size]), found
                    break

    name = _title(name)
    if kind_code is None and "," in text:
        query = text  # an unrecognized qualifier: let OpenWeather interpret it
        label = _title(text)
    elif kind_code is None:
        query, label = name, name
    elif kind_code[0] == "state":
        query, label = f"{name},{kind_code[1]},US", f"{name}, {kind_code[1]}"
    else:
        query, label = f"{name},{kind_code[1]}", f"{name}, {kind_code[1]}"

    qualified = kind_code is not None or "," in text
    ambiguous = [] if qualified else AMBIGUOUS_CITIES.get(name.lower(), [])
    return {"name": name, "query": query, "label": label, "qualified": qualified, "ambiguous": ambiguous}


def same_place_query(a: str, b: str) -> bool:
    """Do two city queries refer to the same place? ('Denver' vs 'Denver,CO,US' -> True)."""
    na, nb = normalize_place(a), normalize_place(b)
    if na["query"].lower().replace(" ", "") == nb["query"].lower().replace(" ", ""):
        return True
    return na["name"].lower() == nb["name"].lower() and not (na["qualified"] and nb["qualified"])


def ambiguity_question(name: str, options: list[tuple[str, str]]) -> str:
    labels = [label for label, _ in options]
    listed = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else " or ".join(labels)
    return f"There's more than one {name}. Did you mean {listed}? Reply with the state or country."


def pick_option(answer: str, options: list[list[str]] | list[tuple[str, str]]) -> tuple[str, str] | None:
    """The option an answer like 'Oregon', 'OR', or 'Portland, Maine' picks, if exactly one."""
    text = _clean(answer)
    picked = []
    for label, query in options:
        region = label.split(",", 1)[1].strip()  # 'Oregon', 'England', 'British Columbia'
        code = query.split(",")[1]  # 'OR', 'GB', 'CA'
        if re.search(rf"\b{re.escape(region)}\b", text, re.IGNORECASE) or re.search(rf"\b{code}\b", text):
            picked.append((label, query))
    return picked[0] if len(picked) == 1 else None
