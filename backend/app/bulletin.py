"""IMD-style text bulletin. Uses Gemini when a key is configured, strictly constrained
to the numbers passed in; otherwise (or on any error) a deterministic template."""
import json

import requests

from .config import GEMINI_API_KEY, GEMINI_MODEL


def _facts(a, impacts):
    c = a["current"]
    return {
        "system": a["name"].title(), "valid_time_utc": a["time"], "position": f"{c['lat']:.1f}N {c['lon']:.1f}E",
        "imd_class": c["imd_class"]["name"], "max_wind_kt_3min": round(c["wind_imd_kt"]),
        "pressure_hpa": round(c["pressure_hpa"]), "movement": f"{c['heading_deg'] or 0:.0f} deg at {c['motion_kmh'] or 0:.0f} km/h",
        "forecast": [{"lead_h": x["lead_h"], "position": f"{x['lat']:.1f}N {x['lon']:.1f}E",
                      "wind_kt": round(x["wind_imd_kt"]), "class": x["imd_class"], "uncertainty_km": round(x["cone_radius_km"])}
                     for x in a["forecast"]["leads"]],
        "rapid_intensification_probability_pct": round(100 * a["ri"]["probability"]),
        "districts": [{"district": d["name"], "state": d["state"], "alert": d["color"], "stage": d["stage"], "eta_h": d["eta_h"]}
                      for d in impacts[:12]],
    }


def template(a, impacts):
    f = _facts(a, impacts)
    lines = [f"PS 26070 PROTOTYPE BULLETIN (illustrative - not an official IMD product)",
             f"Valid: {f['valid_time_utc']} UTC", "",
             f"The {f['imd_class']} {f['system']} lay centred near {f['position']}, moving {f['movement']}. "
             f"Estimated maximum sustained wind {f['max_wind_kt_3min']} kt, central pressure about {f['pressure_hpa']} hPa.", "",
             "Forecast track and intensity:"]
    for x in f["forecast"]:
        lines.append(f"  +{x['lead_h']:>2} h  {x['position']:>13}  {x['wind_kt']:>3} kt  {x['class']:<4}  (+/- {x['uncertainty_km']} km)")
    lines += ["", f"Probability of rapid intensification (>= 30 kt in 24 h): {f['rapid_intensification_probability_pct']}%."]
    if f["districts"]:
        lines += ["", "Districts under alert:"]
        lines += [f"  {d['alert']:<6} {d['stage']:<18} {d['district']} ({d['state']}), closest approach ~{d['eta_h']} h"
                  for d in f["districts"]]
    return "\n".join(lines)


_OFFICIAL = ("INDIA METEOROLOGICAL DEPARTMENT", "RSMC NEW DELHI", "BULLETIN NO", "NATIONAL DISASTER MANAGEMENT")


def _strip_official_headers(text):
    """Belt and braces: drop any line that presents the text as an official agency product."""
    keep = [ln for ln in text.splitlines() if not any(k in ln.upper() for k in _OFFICIAL) or "NOT AN OFFICIAL" in ln.upper()]
    return "\n".join(keep).strip()


def generate(a, impacts):
    if not GEMINI_API_KEY:
        return {"source": "template", "text": template(a, impacts)}
    prompt = ("Write a concise tropical cyclone bulletin in the style of an RSMC New Delhi (IMD) bulletin, in English. "
              "Use ONLY the facts in the JSON below; do not invent numbers, places or times. Start with the line "
              "'PS 26070 PROTOTYPE BULLETIN (illustrative - not an official IMD product)'. Plain text, no markdown.\n\n"
              + json.dumps(_facts(a, impacts), ensure_ascii=False))
    try:
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                          headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048,
                                                     "thinkingConfig": {"thinkingBudget": 0}}}, timeout=25)
        r.raise_for_status()
        text = _strip_official_headers(r.json()["candidates"][0]["content"]["parts"][0]["text"].strip())
        return {"source": f"gemini:{GEMINI_MODEL}", "text": text}
    except Exception as e:  # never let the LLM break the dashboard
        return {"source": "template", "error": str(e)[:200], "text": template(a, impacts)}
