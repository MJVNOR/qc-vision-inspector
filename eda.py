import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # EDA transistor — MVTec binario (good/bad)
    """)
    return


@app.cell
def datos():

    import altair as alt
    import pandas as pd
    from pathlib import Path

    base = Path("transistor_binary")
    labels = pd.read_csv(base / "labels.csv")
    labels

    return alt, base, labels, pd


@app.cell
def tabla(labels, mo):

    labels_table = mo.ui.table(labels, selection="multi", show_column_summaries=False)
    labels_table

    return


@app.cell
def clases(alt, labels, mo):

    class_chart = mo.ui.altair_chart(
        alt.Chart(labels).mark_bar().encode(
            x=alt.X("label:N", title="clase"),
            y=alt.Y("count():Q", title="imagenes"),
            color="label:N",
        ).properties(title="Distribucion good / bad", width=300)
    )
    class_chart

    return


@app.cell
def defectos(alt, labels, mo):

    bad_only = labels[labels.label == "bad"]
    defect_chart = mo.ui.altair_chart(
        alt.Chart(bad_only).mark_bar().encode(
            x=alt.X("defect:N", title="defecto"),
            y=alt.Y("count():Q", title="imagenes"),
            color="defect:N",
        ).properties(title="Defectos dentro de bad (10 c/u esperado)", width=400)
    )
    defect_chart

    return


@app.cell
def muestras(base, labels, mo):

    def _img(label, defect):
        row = labels[(labels.label == label) & (labels.defect == defect)].iloc[0]
        folder = "good" if label == "good" else "bad"
        return mo.image(
            str(base / folder / row["filename"]),
            width=220,
            caption=f"{row['filename']} | {defect} | {row['source']}->{row['original_path']}",
        )

    mo.vstack([
        mo.md("### Muestras: 1 good + 1 por defecto"),
        mo.hstack([_img("good", "good"), _img("bad", "bent_lead"), _img("bad", "cut_lead")]),
        mo.hstack([_img("bad", "damaged_case"), _img("bad", "misplaced")]),
    ])

    return


@app.cell
def tamanos(base, labels, pd):

    from PIL import Image

    _probe = []
    for _, r in labels.iterrows():
        folder = "good" if r["label"] == "good" else "bad"
        p = base / folder / r["filename"]
        w, h = Image.open(p).size
        _probe.append({"filename": r["filename"], "label": r["label"], "w": w, "h": h})
    sizes = pd.DataFrame(_probe)
    sizes[["w", "h"]].describe()

    return (sizes,)


@app.cell
def tamanos_chart(alt, mo, sizes):

    mo.hstack([
        mo.ui.altair_chart(
            alt.Chart(sizes).mark_bar().encode(
                x=alt.X("w:Q", bin=True, title="ancho px"),
                y=alt.Y("count():Q", title="imagenes"),
            ).properties(title="Ancho", width=250)
        ),
        mo.ui.altair_chart(
            alt.Chart(sizes).mark_bar().encode(
                x=alt.X("h:Q", bin=True, title="alto px"),
                y=alt.Y("count():Q", title="imagenes"),
            ).properties(title="Alto", width=250)
        ),
    ])

    return


if __name__ == "__main__":
    app.run()
