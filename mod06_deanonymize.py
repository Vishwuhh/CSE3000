import pandas as pd


import pandas as pd


def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on age, gender, and zip3.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.

    A match is unique only if its combination of quasi-identifiers
    occurs exactly once in EACH dataset.
    """
    keys = ["age", "gender", "zip3"]

    anon_unique = anon_df.loc[
        ~anon_df.duplicated(subset=keys, keep=False)
    ]

    aux_unique = aux_df.loc[
        ~aux_df.duplicated(subset=keys, keep=False)
    ]

    matches = anon_unique.merge(
        aux_unique,
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    return matches[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    return matches_df["anon_id"].nunique() / len(anon_df)

