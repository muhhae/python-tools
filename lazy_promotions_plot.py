import os
from glob import glob
from typing import Final

from numpy import extract
import pandas as pd
from pandas.core import algorithms
from pandas.core.apply import com
import pandasgui

from common import extract_desc, sort_key
import docs_writer
from lazy_promotions_parser import GetResult
from docs_writer import DocsWriter
from plotly_wrapper import Line, Scatter

OUTPUT_PATH: Final[str] = "../lazy_promotions_site/figures/"
DATA_PATH: Final[str] = "../lazy_promotions/"
ALGO_ORDERS = [
    "FIFO",
    "LRU",
    "CLOCK",
    "Offline CLOCK",
    "Q-Clock",
    "Offline Q-CLOCK",
    "QTime-Clock",
    "Offline QTime-Clock",
]


def WriteAggregate(
    writer: DocsWriter,
    df: pd.DataFrame,
    modifier: str,
):
    writer.Write("## Aggregate Results")


def WriteIndividual(
    writer: DocsWriter,
    df: pd.DataFrame,
    modifier: str,
):
    writer.Write("## Individual Results")
    for ign_obj_size in df["Ignore Obj Size"].unique():
        writer.Write(f"### Ignore Object Size = {'True' if ign_obj_size else 'False'}")
        for trace in df["Trace"].unique():
            writer.Write(f"#### {trace}")
            for Y in ["Miss Ratio", "Hit", "Reinserted"]:
                writer.Write(f"##### {Y}")
                for cache_size in df["Cache Size"].unique():
                    data = df.query(
                        "`Trace` == @trace and `Ignore Obj Size` == @ign_obj_size and `Cache Size` == @cache_size"
                    ).sort_values(by="P")
                    fig = Line(
                        data,
                        "P",
                        Y,
                        color="Algorithm",
                        title=f"{trace} # {cache_size * 100}%",
                        category_orders={"Algorithm": ALGO_ORDERS},
                        markers=True,
                    )
                    fig.update_layout(xaxis_dtick=0.125)
                    fig.update_traces(connectgaps=True)
                    writer.WriteFig(fig)
            writer.Write("##### Scatter")
            for cache_size in df["Cache Size"].unique():
                writer.Write(f"###### Cache Size = {cache_size}")
                for p in sorted(df["P"].unique()):
                    data = df.query(
                        "`Trace` == @trace and `Ignore Obj Size` == @ign_obj_size and `Cache Size` == @cache_size and `P` == @p"
                    ).sort_values(by="P")
                    fig = Scatter(
                        data,
                        x="Reinserted",
                        y="Miss Ratio",
                        color="Algorithm",
                        title=f"{trace} # {cache_size * 100}% # P={p}",
                        category_orders={"Algorithm": ALGO_ORDERS},
                        symbol="Algorithm",
                    )
                    writer.WriteFig(fig)


def GenerateSite(
    title: str,
    df: pd.DataFrame,
    modifier: str,
):
    current_title = f"{title}"
    html_path = os.path.join(
        OUTPUT_PATH,
        f"{current_title}.html",
    )
    writer = DocsWriter(html_path=html_path, md_path=None)

    writer.Write(f"# {title}")
    WriteAggregate(writer, df, modifier)
    WriteIndividual(writer, df, modifier)
    writer.Flush()
    print("Finished generating " + current_title)


def main():
    log_path = os.path.join(DATA_PATH, "log")
    files = sorted(glob(os.path.join(log_path, "*.json")), key=sort_key)

    alg: dict[str, str | tuple[str, int]] = {
        "FIFO": "fifo",
        "LRU": "lru",
        "Q-Clock": "q-clock",
        "QTime-Clock": "qtime-clock",
        "CLOCK": ("offline-clock", 0),
        "Offline CLOCK": ("offline-clock", 1),
        "Offline Q-CLOCK": ("offline-q-clock", 1),
        "Offline QTime-Clock": ("offline-qtime-clock", 1),
    }

    dfs: list[pd.DataFrame] = []
    for name, key in alg.items():
        if isinstance(key, str):
            key_files = [f for f in files if key in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name))
        elif isinstance(key, tuple):
            key_files = [f for f in files if key[0] in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name, key[1]))

    combined = pd.concat(dfs)

    combined["P"] = combined["P"].astype("object")
    combined["P"] = combined["P"].apply(
        lambda x: [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        if pd.isna(x)
        else x
    )
    combined = combined.explode("P")

    for group in combined["Trace Group"].unique():
        GenerateSite(group, combined.query("`Trace Group` == @group"), "P")


if __name__ == "__main__":
    main()
