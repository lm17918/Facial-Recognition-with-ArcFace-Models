import pandas as pd
from sklearn.model_selection import train_test_split


def load_annotation_file(annop_path: str, columns: list) -> pd.DataFrame:
    df = pd.read_csv(annop_path, sep="\s+", header=None, skiprows=2, engine="python")
    df.columns = columns
    return df


def merge_annotations_in_dataframe() -> pd.DataFrame:
    attr_annotation_path = "CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
    pose_annotation_path = "CelebAMask-HQ/CelebAMask-HQ-pose-anno.txt"
    attr_annotation_columns: list[str] = [
        "Filename",
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]
    pose_annotation_columns = ["Filename", "Yaw", "Pitch", "Raw"]
    df_attr: pd.DataFrame = load_annotation_file(
        attr_annotation_path, attr_annotation_columns
    )
    df_pose: pd.DataFrame = load_annotation_file(
        pose_annotation_path, pose_annotation_columns
    )
    df = pd.merge(df_attr, df_pose, on="Filename")
    return df


def main():
    """
    Split the original dataset in train, val and test set.
    TODO create a way to stratify the dataset using the informations about the annotations we have reguarding all the people´s face features.
    """
    df = merge_annotations_in_dataframe()

    df["Split"] = "train"
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)

    val["Split"] = "val"
    test["Split"] = "test"
    final_df = pd.concat([train, test, val])

    final_df.to_csv("preprocessed_CelebA.csv", index=False)


if __name__ == "__main__":
    main()
