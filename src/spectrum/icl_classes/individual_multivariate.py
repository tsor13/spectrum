from typing import Any, Dict, List, Optional, Union

import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

# from spectrum.icl_classes.icl_class import ICLClass


class IndividualMultivariate(GenericMultivariate):
    def __init__(
        self,
        df: pd.DataFrame,
        individual_id_column: str,
        given_variables: List[str],
        gen_variables: List[str],
        name: str,
        individual_description_column: str | None = None,
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        # sample_space: Optional[Union[str, List[str]]] = None,
        # data: Optional[pd.DataFrame] = None,
        # data_formatter: Optional[List[DataFormatter]] = None,
    ):
        super().__init__(
            df=df,
            # name=name,
            given_variables=given_variables,
            gen_variables=gen_variables,
            name=name,
            descriptions=descriptions,
            # sample_space=sample_space,
            # data=data,
            # data_formatter=data_formatter,
        )

        # assert that variables are in df
        if individual_id_column not in df.columns:
            raise ValueError(f"{individual_id_column} is not in df")
        if individual_description_column is not None:
            if individual_description_column not in df.columns:
                raise ValueError(f"{individual_description_column} is not in df")

        self.individual_id_column = individual_id_column
        self.individual_description_column = individual_description_column
        # if self.individual_description_column is not None:
        #     raise ValueError("Not yet implemented")

        # get unique individual_ids from df
        individual_ids = df[individual_id_column].unique()
        self.individual_ids = individual_ids
        self.length = len(individual_ids)

    def get_raw_samples(
        self,
        seed: int,
        max_n: int,
        ind: int,
    ) -> List[Dict[str, Any]]:
        self.verify_ind(ind)

        # get samples from for individual
        samples = self.df[
            self.df[self.individual_id_column] == self.individual_ids[ind]
        ].copy()
        # set descriptions if `self.individual_description_column` is not None
        if self.individual_description_column is not None:
            # descriptions has to have length one for the text to be correct
            self.descriptions = [
                samples[self.individual_description_column].tolist()[0]
            ]
        # shuffle
        samples = samples.sample(frac=1, random_state=seed).reset_index(drop=True)
        # get first max_n
        samples = samples.head(max_n)
        # to list of dicts
        samples = samples.to_dict(orient="records")
        return samples
