"""District impact staging and CAP 1.2 XML generation (SACHET-compatible format, illustrative).

Status is always "Exercise": this prototype is not an official IMD/NDMA source.
"""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import numpy as np

from ml.ibtracs import haversine_km

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
IST = timezone(timedelta(hours=5, minutes=30))
_D = json.load(open(os.path.join(os.path.dirname(__file__), "data", "coastal_districts.json"), encoding="utf-8"))
DISTRICTS = [dict(zip(["name", "state", "lang", "lat", "lon"], d)) for d in _D["districts"]]

STAGES = [(24, "Cyclone Warning"), (48, "Cyclone Alert"), (72, "Pre-Cyclone Watch")]
COLOR_ORDER = {"Red": 0, "Orange": 1, "Yellow": 2}
CAP_SEVERITY = {"Red": "Extreme", "Orange": "Severe", "Yellow": "Moderate"}
CAP_URGENCY = {"Cyclone Warning": "Immediate", "Cyclone Alert": "Expected", "Pre-Cyclone Watch": "Future"}

TEXT = {
    "en": {
        "event": "Cyclone",
        "headline": "{name} ({cls}): {color} alert for {n} coastal district(s)",
        "desc": "{name} is at {lat:.1f}N {lon:.1f}E with estimated winds of {wind:.0f} kt ({cls}). "
                "Forecast peak within 72 h: {peak:.0f} kt. Closest approach to affected districts in about {eta} h.",
        "Red": "Take action: move to safe shelter as directed by the district administration. Fishermen must not venture into the sea.",
        "Orange": "Be prepared: secure property, keep emergency kits ready, fishermen should return to the coast.",
        "Yellow": "Be aware: follow official IMD bulletins and local advisories.",
    },
    "hi": {
        "event": "चक्रवात",
        "headline": "चक्रवात {name} ({cls}): {n} तटीय ज़िलों के लिए {color_l} अलर्ट",
        "desc": "चक्रवात {name} {lat:.1f}° उत्तर {lon:.1f}° पूर्व पर है, अनुमानित हवा की गति {wind:.0f} नॉट ({cls})। "
                "अगले 72 घंटों में अधिकतम अनुमानित गति {peak:.0f} नॉट। प्रभावित ज़िलों के सबसे निकट लगभग {eta} घंटे में।",
        "Red": "तुरंत कार्रवाई करें: प्रशासन के निर्देशानुसार सुरक्षित आश्रय में जाएँ। मछुआरे समुद्र में न जाएँ।",
        "Orange": "तैयार रहें: संपत्ति सुरक्षित करें, आपातकालीन सामान तैयार रखें, मछुआरे तट पर लौट आएँ।",
        "Yellow": "सतर्क रहें: IMD के आधिकारिक बुलेटिन और स्थानीय सलाह का पालन करें।",
        "colors": {"Red": "रेड", "Orange": "ऑरेंज", "Yellow": "येलो"},
    },
    "gu": {
        "event": "વાવાઝોડું",
        "headline": "વાવાઝોડું {name} ({cls}): {n} દરિયાકાંઠાના જિલ્લાઓ માટે {color_l} એલર્ટ",
        "desc": "વાવાઝોડું {name} {lat:.1f}° ઉત્તર {lon:.1f}° પૂર્વ પર છે, અંદાજિત પવનની ઝડપ {wind:.0f} નોટ ({cls}). "
                "આગામી 72 કલાકમાં મહત્તમ અંદાજિત ઝડપ {peak:.0f} નોટ. અસરગ્રસ્ત જિલ્લાઓની સૌથી નજીક આશરે {eta} કલાકમાં.",
        "Red": "તરત પગલાં લો: તંત્રની સૂચના મુજબ સુરક્ષિત આશ્રયસ્થાને જાઓ. માછીમારો દરિયામાં ન જાય.",
        "Orange": "તૈયાર રહો: મિલકત સુરક્ષિત કરો, કટોકટીની સામગ્રી તૈયાર રાખો, માછીમારો કિનારે પાછા ફરે.",
        "Yellow": "સાવચેત રહો: IMDના સત્તાવાર બુલેટિન અને સ્થાનિક સૂચનાઓનું પાલન કરો.",
        "colors": {"Red": "રેડ", "Orange": "ઓરેન્જ", "Yellow": "યલો"},
    },
}
LANG_TAG = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN"}


def gale_radius_km(wind_imd):
    """Illustrative radius of damaging winds/rain around the centre."""
    return 40.0 if wind_imd < 34 else float(np.clip(80 + 2.0 * (wind_imd - 34), 60, 250))


def _path(current, leads):
    """Interpolate forecast every 3 h: (hour, lat, lon, cone_radius, wind_imd)."""
    hs = [0] + [x["lead_h"] for x in leads]
    la = [current["lat"]] + [x["lat"] for x in leads]
    lo = [current["lon"]] + [x["lon"] for x in leads]
    rad = [0.0] + [x["cone_radius_km"] for x in leads]
    wi = [current["wind_imd_kt"]] + [x["wind_imd_kt"] for x in leads]
    t = np.arange(0, 73, 3)
    return t, np.interp(t, hs, la), np.interp(t, hs, lo), np.interp(t, hs, rad), np.interp(t, hs, wi)


def district_impacts(a):
    t, la, lo, rad, wi = _path(a["current"], a["forecast"]["leads"])
    impacts = []
    for d in DISTRICTS:
        dist = haversine_km(la, lo, d["lat"], d["lon"])
        reach = rad + np.array([gale_radius_km(w) for w in wi])
        hit = np.where(dist <= reach)[0]
        if not len(hit):
            continue
        i = hit[0]
        eta = int(t[i])
        stage = next(s for h, s in STAGES if eta <= h)
        w = float(wi[hit].max())
        if stage == "Cyclone Warning":
            color = "Red" if w >= 48 else "Orange"
        elif stage == "Cyclone Alert":
            color = "Orange" if w >= 48 else "Yellow"
        else:
            color = "Yellow"
        if w < 28:
            color = "Yellow"
        certainty = "Likely" if dist[hit].min() <= gale_radius_km(w) else "Possible"
        impacts.append({**d, "eta_h": eta, "stage": stage, "color": color, "max_wind_imd_kt": w,
                        "min_distance_km": float(dist.min()), "certainty": certainty})
    impacts.sort(key=lambda x: (COLOR_ORDER[x["color"]], x["eta_h"]))
    return impacts


def build_cap(a, impacts, langs=("en", "hi", "gu")):
    ET.register_namespace("", CAP_NS)
    q = lambda tag: f"{{{CAP_NS}}}{tag}"  # noqa: E731
    root = ET.Element(q("alert"))
    sent = datetime.now(IST).replace(microsecond=0)
    for tag, val in [("identifier", f"PS26070-{a['sid']}-{uuid.uuid4().hex[:8]}"), ("sender", "ps26070-prototype@localhost"),
                     ("sent", sent.isoformat()), ("status", "Exercise"), ("msgType", "Alert"), ("scope", "Public"),
                     ("note", "Illustrative CAP output from the PS 26070 prototype. Not an official IMD/NDMA warning.")]:
        ET.SubElement(root, q(tag)).text = val
    if not impacts:
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    cur, leads = a["current"], a["forecast"]["leads"]
    peak = max([cur["wind_imd_kt"]] + [x["wind_imd_kt"] for x in leads])
    groups = {}
    for d in impacts:
        groups.setdefault(d["color"], []).append(d)
    top = min(groups, key=COLOR_ORDER.get)
    top_stage = min((d for d in impacts if d["color"] == top), key=lambda d: d["eta_h"])["stage"]
    cone = a["forecast"]["cone"]
    polygon = " ".join(f"{lat:.3f},{lon:.3f}" for lon, lat in cone)

    for lang in langs:
        tx = TEXT[lang]
        info = ET.SubElement(root, q("info"))
        fmt = dict(name=a["name"].title(), cls=cur["imd_class"]["code"], n=len(impacts), color=top,
                   color_l=tx.get("colors", {}).get(top, top), lat=cur["lat"], lon=cur["lon"],
                   wind=cur["wind_imd_kt"], peak=peak, eta=min(d["eta_h"] for d in impacts))
        for tag, val in [("language", LANG_TAG[lang]), ("category", "Met"), ("event", tx["event"]),
                         ("responseType", "Prepare" if top != "Red" else "Evacuate"),
                         ("urgency", CAP_URGENCY[top_stage]), ("severity", CAP_SEVERITY[top]),
                         ("certainty", "Likely" if any(d["certainty"] == "Likely" for d in impacts) else "Possible"),
                         ("effective", sent.isoformat()), ("expires", (sent + timedelta(hours=6)).isoformat()),
                         ("senderName", "PS 26070 prototype (not an official IMD/NDMA source)"),
                         ("headline", tx["headline"].format(**fmt)), ("description", tx["desc"].format(**fmt)),
                         ("instruction", tx[top])]:
            ET.SubElement(info, q(tag)).text = val
        for name, value in [("ColorCode", top), ("IMDClass", cur["imd_class"]["code"]),
                            ("ForecastMethod", a["forecast"]["method"])]:
            p = ET.SubElement(info, q("parameter"))
            ET.SubElement(p, q("valueName")).text = name
            ET.SubElement(p, q("value")).text = value
        for color in sorted(groups, key=COLOR_ORDER.get):
            area = ET.SubElement(info, q("area"))
            ET.SubElement(area, q("areaDesc")).text = f"{color}: " + ", ".join(
                f"{d['name']} ({d['state']})" for d in groups[color])
            ET.SubElement(area, q("polygon")).text = polygon
            for d in groups[color]:
                g = ET.SubElement(area, q("geocode"))
                ET.SubElement(g, q("valueName")).text = "District"
                ET.SubElement(g, q("value")).text = f"{d['state']}/{d['name']}"
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
