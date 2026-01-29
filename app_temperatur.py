import re
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# How to run the app
# python3 -m streamlit run app_temperatur.py 
# -----------------------------

PUNKT = "."
START = "<START>"

TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+|[.!?]")


def text_zu_woertern(text: str) -> List[str]:
    """Text -> Wörterliste. ! und ? zählen wie ein Punkt."""
    teile = TOKEN_RE.findall(text)
    woerter = []
    for t in teile:
        if t in (".", "!", "?"):
            woerter.append(PUNKT)
        else:
            woerter.append(t.lower())
    return woerter


def baue_uebergaenge(
    woerter: List[str],
    anzahl_vorher: int
) -> Tuple[Dict[Tuple[str, ...], Counter], Dict[Tuple[str, ...], int]]:
    """
    anzahl_vorher:
      1 -> 2-Wörter-Modell: nächstes Wort hängt vom letzten Wort ab
      2 -> 3-Wörter-Modell: nächstes Wort hängt von den letzten zwei Wörtern ab
    """
    start = tuple([START] * anzahl_vorher)

    # START am Anfang und nach jedem Punkt
    seq = list(start)
    for w in woerter:
        seq.append(w)
        if w == PUNKT:
            seq.extend(start)

    uebergaenge: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    gesamt: Dict[Tuple[str, ...], int] = defaultdict(int)

    for i in range(anzahl_vorher, len(seq)):
        vorher = tuple(seq[i - anzahl_vorher:i])
        naechstes = seq[i]
        uebergaenge[vorher][naechstes] += 1
        gesamt[vorher] += 1

    return uebergaenge, gesamt


def tabelle_bauen(
    uebergaenge: Dict[Tuple[str, ...], Counter],
    gesamt: Dict[Tuple[str, ...], int]
) -> pd.DataFrame:
    """Tabelle: Vorher -> Nächstes Wort -> Wahrscheinlichkeit."""
    zeilen = []
    for vorher, cnt in uebergaenge.items():
        total = gesamt[vorher]
        for naechstes, anzahl in cnt.items():
            zeilen.append({
                "Vorher": " ".join(vorher),
                "Nächstes Wort": naechstes,
                "Wahrscheinlichkeit": anzahl / total if total else 0.0
            })
    df = pd.DataFrame(zeilen)
    if not df.empty:
        df = df.sort_values(["Vorher", "Wahrscheinlichkeit"], ascending=[True, False]).reset_index(drop=True)
    return df


def waehle_naechstes(cnt: Counter, zufall: int, rng: random.Random) -> str:
    """
    zufall:
      0   -> immer das wahrscheinlichste Wort
      100 -> komplett zufällig (alle möglichen nächsten Wörter gleich wahrscheinlich)
      dazwischen -> Mischung
    """
    moeglich = list(cnt.keys())
    if not moeglich:
        return PUNKT

    # 0% Zufall
    if zufall <= 0:
        best = max(cnt.values())
        beste_woerter = [w for w, c in cnt.items() if c == best]
        return rng.choice(beste_woerter)

    # 100% Zufall
    if zufall >= 100:
        return rng.choice(moeglich)

    # Mischung über Gewichte:
    # alpha = Zufall/100
    # Gewicht = (1-alpha)*Häufigkeit + alpha*1
    alpha = zufall / 100.0
    weights = []
    for w in moeglich:
        haeufigkeit = cnt[w]
        gewicht = (1 - alpha) * haeufigkeit + alpha * 1
        weights.append(gewicht)

    return rng.choices(moeglich, weights=weights, k=1)[0]


def satzanfang_zu_start(
    satzanfang: str,
    anzahl_vorher: int
) -> Tuple[Tuple[str, ...], List[str], str]:
    """
    Satzanfang -> Start-Zustand.
    Gibt (start, bereits_ausgabe, fehlermeldung) zurück.
    fehlermeldung ist "" wenn alles ok ist.
    """
    if not satzanfang.strip():
        return tuple([START] * anzahl_vorher), [], ""

    woerter = satzanfang.lower().strip().split()

    if "." in woerter:
        return tuple([START] * anzahl_vorher), [], "Im Satzanfang ist ein Punkt nicht erlaubt."

    if len(woerter) != anzahl_vorher:
        return (
            tuple([START] * anzahl_vorher),
            [],
            f"Für dieses Modell musst du genau {anzahl_vorher} Wort/Wörter als Satzanfang eingeben."
        )

    return tuple(woerter), woerter.copy(), ""


def satz_erzeugen(
    uebergaenge: Dict[Tuple[str, ...], Counter],
    anzahl_vorher: int,
    zufall: int,
    max_woerter: int,
    rng: random.Random,
    satzanfang: str = ""
) -> str:
    start, ausgabe, fehler = satzanfang_zu_start(satzanfang, anzahl_vorher)
    if fehler:
        return ""

    vorher = start

    for _ in range(max_woerter):
        cnt = uebergaenge.get(vorher)
        if not cnt:
            vorher = start
            cnt = uebergaenge.get(vorher, Counter({PUNKT: 1}))

        naechstes = waehle_naechstes(cnt, zufall, rng)

        # Satz darf nicht mit Punkt starten
        if not ausgabe and naechstes == PUNKT:
            vorher = start
            continue

        ausgabe.append(naechstes)

        # Fenster aktualisieren
        if anzahl_vorher == 1:
            vorher = (naechstes,)
        else:
            vorher = (vorher[-1], naechstes)

        if naechstes == PUNKT:
            break

    # Wenn kein Punkt kam: Punkt anhängen
    if ausgabe and ausgabe[-1] != PUNKT:
        ausgabe.append(PUNKT)

    # Punkt ohne Leerzeichen davor
    s = []
    for w in ausgabe:
        if w == PUNKT:
            if s:
                s[-1] = s[-1] + PUNKT
            else:
                s.append(PUNKT)
        else:
            s.append(w)
    return " ".join(s)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Eigenes Sprachmodell", layout="wide")
st.title("Eigenes Sprachmodell")

st.info(
    "Hier kannst du dein eigenes einfaches Sprachmodell aufbauen. "
    "Es analysiert deinen Text und bestimmt, welche Wörter typischerweise aufeinander folgen. "
    "Anschließend kann dein Modell neue Sätze erzeugen, indem es diese Übergänge nutzt."
)

st.subheader("1. Text eingeben")
text = st.text_area(
    "Dein Trainings-Text",
    value=(
        "Das ist ein Beispieltext. "
        "Du kannst hier deinen Text einfügen. "
        "Der Generator lernt, welche Wörter oft aufeinander folgen. "
        "Wenn ein Satz endet, fängt ein neuer an. " 
    ),
    height=220,
)

with st.sidebar:
    st.header("Einstellungen")

    modell = st.radio(
        "Kontextlänge",
        ["2 Wörter", "3 Wörter"],
        help=(
            "Die Kontextlänge gibt an, wie viele vorherige Wörter "
            "bei der Auswahl des nächsten Wortes berücksichtigt werden."
        )
    )

    zufall = st.slider(
        "Zufall",
        0, 100, 20, 1,
        help="0 = wahrscheinlichstes Wort, 100 = komplett zufällig"
    )

    st.markdown(
        "<small>"
        "Der Zufall entscheidet nur zwischen Wörtern,"
        "die im Text an dieser Stelle vorkommen.<br>"
        "α = Zufall / 100<br>"
        "Gewicht = (1 − α) · Häufigkeit + α"
        "</small>",
        unsafe_allow_html=True
    )


    anzahl_vorher = 1 if modell == "2 Wörter" else 2
    placeholder = "z. B. der" if anzahl_vorher == 1 else "z. B. der hund"

    satzanfang_text = st.text_input(
        "Satzanfang (optional)",
        placeholder=placeholder,
        help=(
            "Gibt einen festen Start für den Satz vor.\n"
            "Beim 2-Wörter-Modell: 1 Wort.\n"
            "Beim 3-Wörter-Modell: 2 Wörter."
        )
    )

    anzahl_saetze = st.number_input(
        "Anzahl Sätze",
        min_value=1, max_value=50, value=5, step=1
    )

    max_woerter = st.number_input(
        "Maximale Wörter pro Satz",
        min_value=3, max_value=60, value=25, step=1,
        help="Sicherheitsgrenze, falls kein Punkt gewählt wird."
    )


# Intern: feste Zufallsquelle (keine Extra-Einstellung für Schüler)
rng = random.Random(1)

woerter = text_zu_woertern(text)
uebergaenge, gesamt = baue_uebergaenge(woerter, anzahl_vorher)

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("2. Übergangswahrscheinlichkeiten")

    df = tabelle_bauen(uebergaenge, gesamt)
    if df.empty:
        st.warning("Bitte mehr Text eingeben.")
    else:
        suche_vorher = st.text_input(
            "Suche (nur „Vorher“)",
            placeholder=placeholder,
        )

        df_anzeige = df

        if suche_vorher.strip():
            teile = suche_vorher.lower().strip().split()

            if "." in teile:
                st.error("In der Suche ist ein Punkt nicht erlaubt. Bitte nur Wörter eingeben.")
                df_anzeige = df.iloc[0:0]

            elif len(teile) != anzahl_vorher:
                st.error(
                    f"Für dieses Modell musst du genau {anzahl_vorher} Wort/Wörter eingeben "
                    f"(du hast {len(teile)} eingegeben)."
                )
                df_anzeige = df.iloc[0:0]

            else:
                gesucht = " ".join(teile)
                df_anzeige = df[df["Vorher"] == gesucht]
                if df_anzeige.empty:
                    st.warning(
                        f"„{gesucht}“ kommt als „Vorher“ im Text nicht vor. "
                        "Versuche andere Wörter oder mehr Trainings-Text."
                    )

        st.dataframe(df_anzeige, use_container_width=True, height=520)

        # Hinweis, warum Zufall manchmal nichts ändert
        moeglichkeiten = df.groupby("Vorher")["Nächstes Wort"].nunique()
        nur_eine = int((moeglichkeiten == 1).sum())
        gesamt_vorher = int(len(moeglichkeiten))
        st.caption(
            f"Info: Bei {nur_eine} von {gesamt_vorher} Fällen gibt es nur **ein** mögliches nächstes Wort. "
            "Dann kann der Zufall-Regler nichts ändern."
        )

with colB:
    st.subheader("3. Sätze erzeugen")

    if st.button("Sätze erzeugen"):
        # Satzanfang prüfen und ggf. Fehlermeldung zeigen
        _, _, fehler = satzanfang_zu_start(satzanfang_text, anzahl_vorher)
        if fehler:
            st.error(fehler)
        else:
            for i in range(int(anzahl_saetze)):
                s = satz_erzeugen(
                    uebergaenge,
                    anzahl_vorher,
                    int(zufall),
                    int(max_woerter),
                    rng,
                    satzanfang_text
                )
                st.write(f"{i+1}. {s}")
