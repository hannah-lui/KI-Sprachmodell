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

# -----------------------------
# Session State
# -----------------------------
if "generated_sentences" not in st.session_state:
    st.session_state.generated_sentences = []

if "last_model" not in st.session_state:
    st.session_state.last_model = None


st.info(
    "Hier kannst du dein eigenes kleines Sprachmodell aufbauen. "
    "Es analysiert deinen Text und bestimmt für dich die Übgergangstabelle. "
    "Mit den Parametern kannst du varieren, wie dein Modell arbeitet."
)

st.subheader("1. Trainingstext")
text = st.text_area(
    "Du kannst den Text beliebig verändern. Ergänze noch die Liedtexte von anderen Liedern. Wichtig ist, dass ein . am Satzende steht.",
    value=(
        "Ich glaub, ich will heut nicht mehr geh'n. Ich hab dich viel zu kurz geseh'n. Und überhaupt, draußen ist's kalt, zu kalt. Und an dein'n Fensterecken blüht das Eis. Ich will, dass du gar nichts machst, gar nichts machst. Will mit dir den ganzen Tag, sag alles ab. In meinem Bett ist so viel Platz und mir ist kalt. Doch auf deinem Screen ist Portugal. Denn du willst viel vom Leben, glaub an dich. Ey, deine Ziele sind zu ambitioniert für mich. Du inspirierst, doch bitte sei heute faul für mich. All meine Freunde unterwegs, suchen, was wir schon sind. Ich glaub, ich will heut nicht mehr geh'n. Ich hab dich viel zu kurz geseh'n. Und überhaupt, draußen ist's kalt. Und an dein'n Fensterecken blüht das Eis. Ich glaub, ich will nie mehr nach Haus. Weil da, wo du bist, ist das auch . Komm, schließ uns ein, wir sind allein. Frag mich, ob das Gefühl für immer bleibt. Palo Santo, Herz in Brand, mach die teuren Kerzen an. Frisch geduscht, Bossa nova, verlier'n uns im Viervierteltakt. Ich les dir von den Lippen ab, weil du nur gute Seiten hast. Sitz auf deiner Fensterbank und strahl heut alle Sterne an. Ich glaub an mich, glaub an dich. Eigentlich nur wir, ansonsten brauch ich nichts, brauch nur dich. Mach keine Pläne, bitte sei heute faul für mich. All meine Leute unterwegs, suchen, was wir schon sind. Ich glaub, ich will heut nicht mehr geh'n. Ich hab dich viel zu kurz geseh'n. Und überhaupt, draußen ist's kalt, zu kalt. Und an dein'n Fensterecken blüht das Eis. Ich glaub, ich will heut nicht mehr geh'n. Ich hab dich viel zu kurz geseh'n. Und überhaupt, draußen ist's kalt. Und an dein'n Fensterecken blüht das Eis. Ich glaub, ich will nie mehr nach Haus. Weil da, wo du bist, ist das auch. Komm, schließ uns ein, wir sind allein. Frag mich, ob das Gefühl für immer bleibt. Ich glaub, ich will heut nicht mehr geh'n."
        "Ich will, dass ihr mir vertraut. Ich will, dass ihr mir glaubt. Ich will eure Blicke spüren. Jeden Herzschlag kontrollieren. Ich will eure Stimmen hören. Ich will die Ruhe stören. Ich will, dass ihr mich gut seht. Ich will, dass ihr mich versteht. Ich will eure Fantasie. Ich will eure Energie. Ich will eure Hände seh'n. Könnt ihr mich hören?  Könnt ihr mich seh'n? Könnt ihr mich fühlen? Ich versteh' euch nicht. Könnt ihr uns hören? Könnt ihr uns seh'n? Könnt ihr uns fühlen? Wir versteh'n euch nicht."
        "Ich hab′ dich lieb, so lieb, lieber als je zuvor. Ich hab' dich lieb, so lieb, ich nehm's halt mit Humor. Ich hab′ dich lieb, so lieb, lieber als je zuvor. Ich hab′ dich lieb, so lieb, ich nehm's halt mit Humor. Du wolltest dich nicht an mich binden, bin ich so ′n oller Baum. Eine Familie mit dir, das war mein Traum, doch dir war's viel zu früh. Es gibt bestimmt doch noch bessere, andere als mich. Du willst dich erst umsehen, man weiß ja nie. Doch ich hab′ dich lieb, so lieb, lieber als du denkst. Ich hab' dich lieb, so lieb, auch wenn du nicht an mir hängst. Ruf doch mal wieder an und erzähl mir, was du träumst. Ist ganz egal, wann und überrasch mich, komm her und sag, dass du bleibst. Für immer jetzt, für ewig, oder mehr. Doch Halt, ich muss wohl schon träumen, jeder hat so seinen Tick. Für deine Suche wünsch′ ich dir viel Glück. Ich hab' dich lieb, so lieb, ich hoffe, du verzeihst. Ich hab' dich lieb, so lieb, ich will nur, dass du es weißt. Ich hab′ dich lieb, so lieb, lieber als du denkst. Ich hab′ dich lieb, so lieb, auch wenn du nicht an mir hängst."
        "Wenn wir nachts nach Hause gehen, die Lippen blau vom Rotwein. Und wir uns bis vorne an der Ecke meine große Jacke teilen. Der Himmel wird schon morgenrot, doch du willst noch nicht schlafen. Ich hole uns die alten Räder und wir fahren zum Hafen. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich hör mit dir Platten, die ich nicht mag. Ich bin für dich leise, wenn du zu laut bist. Renn für dich zum Kiosk, ob Nacht oder Tag. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich schaue mir Bands an, die ich nicht mag. Ich gehe mit dir in die schlimmsten Schnulzen. Ist mir alles egal, hauptsache du bist da.Ich würde meine Lieblingsplatten sofort für dich verbrennen. Und wenn es für dich wichtig ist, bis nach Barcelona trampen.Die Morgenluft ist viel zu kalt und ich werde langsam heiser. Ich seh nur dich im Tunnelblick, und die Stadt wird langsam leiser. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich hör mit dir Platten, die ich nicht mag. Ich bin für dich leise, wenn du zu laut bist. Renn für dich zum Kiosk, ob Nacht oder Tag. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich schaue mir Bands an, die ich nicht mag. Ich gehe mit dir in die schlimmsten Schnulzen. Ist mir alles egal, hauptsache du bist da. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich hör mit dir Platten, die ich nicht mag. Ich bin für dich leise, wenn du zu laut bist. Renn für dich zum Kiosk, ob Nacht oder Tag. Ich lass für dich das Licht an, obwohl's mir zu hell ist. Ich schaue mir Bands an, die ich nicht mag. Ich gehe mit dir in die schlimmsten Schnulzen. Ist mir alles egal, hauptsache du bist da. Wenn wir Nachts nach Hause gehen. Die Lippen blau vom Rotwein. Und wir uns bis vorne an der Ecke. Meine große Jacke teilen."
        "Und nach dem Abendessen sagte er. Lass mich noch eben Zigaretten holen gehen. Sie rief ihm nach: Nimm dir die Schlüssel mit. Ich werd' inzwischen nach der Kleinen sehen. Er zog die Tür zu, ging stumm hinaus ins neon-helle Treppenhaus. Es roch nach Bohnerwachs und Spießigkeit. Und auf der Treppe dachte er wie wenn das jetzt ein Aufbruch wär'. Ich müsste einfach gehen für alle Zeit. Ich war noch niemals in New York. Ich war noch niemals auf Hawaii. Ging nie durch San Francisco in zerrissenen Jeans. Ich war noch niemals in New York. Ich war noch niemals richtig frei. Einmal verrückt sein und aus allen Zwängen fliehen. Und als er draußen auf der Straße stand. Fiel ihm ein, dass er fast alles bei sich trug den Pass, die Euro checks und etwas Geld. Vielleicht ging heute Abend noch ein Flug. Er könnt' ein Taxi nehmen dort am Eck oder Autostop und einfach weg. Die Sehnsucht in ihm wurde wieder wach. Noch einmal voll von Träumen sein, sich aus der Enge hier befreien. Er dachte über seinen Aufbruch nach. Ich war noch niemals in New York. Ich war noch niemals auf Hawaii. Ging nie durch San Francisco in zerrissenen Jeans. Ich war noch niemals in New York. ich war noch niemals richtig frei. Einmal verrückt sein und aus allen Zwängen fliehen."
   
   ),
    height=220,
)

with st.sidebar:
    st.header("Einstellungen")

    modell = st.radio(
        "Kontextlänge",
        ["2-Gram", "3-Gram"],
        help=(
            "Die Kontextlänge gibt an, wie viele vorherige Wörter "
            "bei der Auswahl des nächsten Wortes berücksichtigt werden."
        )
    )

    # Wenn Modell gewechselt wurde: generierte Sätze löschen
    if st.session_state.last_model is None:
        st.session_state.last_model = modell
    elif st.session_state.last_model != modell:
        st.session_state.generated_sentences = []
        st.session_state.last_model = modell


    zufall = st.slider(
        "Temperatur",
        0, 100, 20, 1,
        help="0 = wahrscheinlichstes Wort, 100 = komplett zufällig"
    )

    st.markdown(
        "<small>"
        "Die Temperatur legt fest, wie stark Zufall bei der Wortwahl einfließt.<br>"
        "α = Temperatur / 100<br>"
        "Score = (1 − α) · Wahrscheinlichkeit + α"
        "</small>",
        unsafe_allow_html=True
    )


    anzahl_vorher = 1 if modell == "2-Gram" else 2
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
    st.subheader("2. Übergangstabelle")

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
            # Nur hier löschen/neu erzeugen:
            st.session_state.generated_sentences = []

            for i in range(int(anzahl_saetze)):
                s = satz_erzeugen(
                    uebergaenge,
                    anzahl_vorher,
                    int(zufall),
                    int(max_woerter),
                    rng,
                    satzanfang_text
                )
                st.session_state.generated_sentences.append(s)

    # Anzeige: bleibt bei Suche/sonstigen Änderungen erhalten
    if st.session_state.generated_sentences:
        for i, s in enumerate(st.session_state.generated_sentences, start=1):
            st.write(f"{i}. {s}")

st.markdown("---")
st.caption(
    "Der vorgelegte Trainingstext setzt sich aus Ausschnitten folgender Lieder zusammen: "
    "„Ich glaub ich will heut nicht mehr“ (Nina Chuba und Provinz), "
    "„Ich will“ (Rammstein), "
    "„Ich hab dich lieb“ (Herbert Grönemeyer), "
    "„Ich lass für dich das Licht an“ (Revolverheld) "
    "und „Ich war noch niemals in New York“ (Udo Jürgens). "
    "Die Texte werden ausschließlich zu Demonstrations- und Lernzwecken verwendet."
)

