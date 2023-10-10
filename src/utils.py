import pandas as pd
import re
from typing import List
import json


def get_parents_ids(steel_df, gspt2gspt_path):
    """Melt ownership information from each plant.
    Ex: 
    Company_A 50%; Company_B 30%, Company_C 20% ->
    
    [Parent, Share]
    [Company_A, 0.5]
    [Company_B, 0.3] 
    [Company_C, 0.2]

    Args:
        steel_df (pd.DataFrame): raw steel data
        gspt2gspt_path (str): path to json gspt2gspt dict

    Returns:
        pd.DataFrame: 
    """
    assert steel_df.columns.isin(["Plant ID", "Parent PermID", "Parent"]).sum() == 3
    melt_df2 = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(), parents_col=steel_df["Parent"]
    )

    melt_df = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(),
        parents_col=steel_df["Parent PermID"],
    )

    mergedf = pd.concat([melt_df2, melt_df], axis=1)

    # TODO: quick fix bc duplicated Plant ID and Share col
    mergedf = mergedf.loc[:, ~mergedf.columns.duplicated()].copy()
    # reorder cols
    mergedf = mergedf.loc[:, ["Parent PermID", "Parent", "Plant ID", "Share"]]
    
    # standardise parent names (avoid multiple entities for same company)
    with open(gspt2gspt_path) as f:
        gspt2gspt = json.load(f)
    
    mergedf["Parent"] = mergedf['Parent'].replace(gspt2gspt)
    
    return mergedf


def split_parent_shares(plant_ids_list: List, parents_col: pd.Series) -> pd.DataFrame:
    """
    Steel plants may be owned by multiple parent companies, with non equally distributed shares.
    Emissions should be attributed to individual companies according to those weights.

    Args:
        parents_list (List[dict]): parents_list[i] = {"comp_A": ownership share for plant i}
        emissions_list (_type_): [emissions_1 -> plant_1 emissions, emissions_2 -> plant_2 emissions]

    Returns:
        melted_parents_df (pd.DataFrame): ["Plant ID", "Parent", "Share"]
    """
    assert len(plant_ids_list) == len(parents_col)
    n_plants = len(set(plant_ids_list))
    
    parents_list = []
    for plant_id, parents in zip(plant_ids_list, parents_col.str.split("; ").tolist()):
        for name_share in parents:
            plantid_parent_share = clean_col(plant_id, name_share)
            parents_list.extend(plantid_parent_share)

    melted_parents_df = pd.DataFrame(
        parents_list, columns=["Plant ID", parents_col.name, "Share"]
    )
    assert len(melted_parents_df['Plant ID'].unique()) == n_plants
    assert melted_parents_df.notna().all().all()

    # Normalise company names
    to_rename = {
        "other": "Other",
        "Ansteel Group Corporation": "Ansteel Group Corp Ltd",
        "Baoshan Iron & Steel Co.,Ltd.": "Baoshan Iron & Steel Co Ltd",
        "Baoshan Iron and Steel Co., Ltd.": "Baoshan Iron & Steel Co Ltd",
        "Hunan Valin Steel Co.,Ltd.": "Hunan Valin Steel Co Ltd"
    }
    melted_parents_df.replace({"Parent": to_rename}, inplace=True)
    return melted_parents_df.reset_index(drop=True)


def clean_col(plant_id: str, name_share: str):
    # TODO: reprendre cette fonction et celles du dessus 
    if plant_id in ["SCN00175", 'SCN00175-1']:
        # 'Ansteel Group Corporation 42.67%, China Minmetals Corporation 6.96%, 
        # Benxi Beifang Investment Co., Ltd. 7.54%, other  42.83%
        id_parent_share = [(plant_id, "Ansteel Group Corporation", 0.4267),
                           (plant_id, "China Minmetals Corporation", 0.0696),
                           (plant_id, "Benxi Beifang Investment Co., Ltd.", 0.0754),
                           (plant_id, "Other", 0.4283)]   
    elif plant_id == "SCN00096":
        id_parent_share = [(plant_id, "Beijing Jianlong Investment Co., Ltd.", 0.7184),
            (plant_id, "Fosun Holdings", 0.249),
            (plant_id, "Other", 0.0326)]   
    else:
        id_parent_share = [(plant_id, None, None)]
        # retrieve string before first square bracket
        names = re.findall("(.*?)\s*\[", name_share)[0]

        # retrieve number between square brackets
        shares = re.findall(r"[-+]?(?:\d*\.*\d+)%", name_share)
        shares = [float(s[:-1]) / 100 for s in shares]

        # some strings before square brackets contain
        # all the information. Second split required
        names = re.split("\s(\d+%,\s|\d+%)", names)
        if len(names) > 1:
            # TODO: ça fonctionne
            names = [
                name for name in names if bool(name) if not bool(re.search("\d+%", name))
            ]
            shares = shares[:-1]
            assert len(names) == len(shares)

        if len(names) == len(shares) == 1:
            id_parent_share = [(plant_id, names[0], shares[0])]
        elif (len(names) > 1) & (len(shares) > 1):
            repeat_plant_id = [plant_id] * len(names)
            id_parent_share = list(zip(repeat_plant_id, names, shares))

    return id_parent_share


if __name__ == "__main__":
    from pathlib import Path
    project_dir = Path().cwd()
    path = project_dir / "data/intermediate/basic_steel_data.xlsx"
    steel_df = pd.read_excel(path)
    capacity_col = "Nominal crude steel capacity (ttpa)"
    # todo: what can we do with missing capacity values ? not much
    steel_df = steel_df.loc[steel_df[capacity_col].notna()]

    # test
    melt_df2 = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(), parents_col=steel_df["Parent"]
    )

    melt_df = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(),
        parents_col=steel_df["Parent PermID"],
    )

    # tests
    plant_count2 = melt_df2.groupby("Plant ID").size().reset_index()
    plant_count = melt_df.groupby("Plant ID").size().reset_index()
    mergedf = pd.merge(
        plant_count2,
        plant_count,
        on="Plant ID",
        how="outer",
        suffixes=("_parent", "_permid"),
    )
    mergedf["diff"] = (mergedf["0_parent"] - mergedf["0_permid"]).abs()

    # TODO: 4 parent companies, 1 permID for Plant ID SCN00116

    assert melt_df.Share.between(0, 1).all()
    # assert (melt_df.groupby(["Parent PermID", "Plant ID"])["Share"] == 1).all()
    assert not melt_df.duplicated(subset=["Plant ID", "Parent"]).any()
