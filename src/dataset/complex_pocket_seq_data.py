from torch.utils.data import Dataset
from .complex_only_data import ComplexOnlyData
from .global_pocket_seq_data import GlobalPocketSeqDataset
from .embedding.embedding import EmbeddingModel

class ComplexPocketSeqDataset:
    """Build complex-only, global-pocket-seq, or complex-pocket-seq datasets."""

    def __init__(
        self,
        column_keys,
        csv_paths,
        embedding_config,
        tmp_dir,
        complex_pocket_folder,
        protein_only_folder,
        ligand_folder,
        work_path,
        if_reload,
        merge_sdf_pdb,
        node_featurizer,
        edge_featurizer,
        get_valid_item,
        assign_label2x,
        filter_ts_valid,
        use_ts_reaction,
        istrain=True,
        n_jobs=1,
        dataset_mode="complex_pocket_seq",  # or "global_pocket_seq" / "complex_only"
        **kwargs,
    ):
        self.dataset_mode = dataset_mode

        self.protein_id = column_keys.get("protein_id", None)
        self.substrate_id = column_keys.get("substrate_id", None)
        self.label_key = column_keys.get("label_key", None)
        self.pic50_key = column_keys.get("pic50_key", None)
        self.sites_key = column_keys.get("sites_key", None)
        self.sequence_key = column_keys.get("sequence_key", None)
        self.reaction_key = column_keys.get("reaction_key", None)

        self.train_csv_path = csv_paths.get("train_csv_path", None)
        self.valid_csv_path = csv_paths.get("valid_csv_path", None)
        self.inference_csv_path = csv_paths.get("inference_csv_path", None)
        self.reaction_csv_path = csv_paths.get("reaction_csv_path", None)
        self.substrate_csv_path = csv_paths.get("substrate_csv_path", None)
        self.product_csv_path = csv_paths.get("product_csv_path", None)
        self.sequence_csv_path = csv_paths.get("sequence_csv_path", None)

        self.embedding_save_dir_base = embedding_config.get("save_dir_base", None)
        self.dataset_embedding_name = embedding_config.get("dataset_name", None)
        self.sequence_embedding_output = embedding_config.get("sequence_output", "sequence_em")
        self.load_substrate_embedding = embedding_config.get("load_substrate_embedding", False)
        self.load_product_embedding = embedding_config.get("load_product_embedding", False)
        self.load_sequence_embedding = embedding_config.get("load_sequence_embedding", False)
        self.load_pocket_embedding = embedding_config.get("load_pocket_embedding", False)
        self.load_reaction_fp_embedding = embedding_config.get("load_reaction_fp_embedding", False)
        self.load_reaction_embedding = embedding_config.get("load_reaction_embedding", False)
        self.use_product_embedding = self.load_product_embedding
        self.use_pocket_embedding = self.load_pocket_embedding
        self.use_pocket_aa_embedding = False
        self.fingerprint = embedding_config.get("fingerprint", {})
        self.base_tmp_dir = tmp_dir
        self.complex_pocket_folder = complex_pocket_folder
        self.protein_only_folder = protein_only_folder
        self.ligand_folder = ligand_folder
        self.energy_output_folder = None
        self.work_path = work_path
        self.if_reload = if_reload
        self.filter_ts_valid = filter_ts_valid
        self.use_ts_reaction = use_ts_reaction

        self.merge_sdf_pdb = merge_sdf_pdb
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.assign_label2x = assign_label2x
        self.get_valid_item = get_valid_item
        self.sequence_pocket_aa_em = None
        self.sequence_pocket_aa_mask = None
        self.sequence_pocket_aa_label = None
        self.sequence_pocket_aa_valid = None
        self.sequence_pocket_active_sites = None

        # Debug row limit.
        self.debug_n = kwargs.get("debug_n", None)
        self.n_jobs = n_jobs

        print(f"[{self.__class__.__name__}] Initializing Embedding Model...")
        self.embeddingmodel = EmbeddingModel(tmp_dir, work_path)

        (
            self.ts_reaction_em,
            self.substrate_em,
            self.sequence_em,
            self.pocket_em,
            self.product_em,
            self.reaction_fp_em,
        ) = self.get_embedding_parts(
            self.reaction_csv_path,
            self.substrate_csv_path,
            self.sequence_csv_path,
            self.product_csv_path,
        )

        self.complex_only_data = None
        self.global_pocket_seq_data = None
        self.conformation_all_df_list = []

        if self.dataset_mode in {"complex_only", "complex_pocket_seq"}:
            self.complex_only_data = ComplexOnlyData(
                istrain=istrain,
                n_jobs=n_jobs,
                shared_context=self,
                **kwargs,
            )
            self.conformation_all_df_list = list(self.complex_only_data.all_df_list)

        if self.dataset_mode == "global_pocket_seq":
            self.global_pocket_seq_data = GlobalPocketSeqDataset(self, istrain=istrain, existing_df=None)
        elif self.dataset_mode == "complex_pocket_seq":
            merged_existing_df = None
            if self.conformation_all_df_list:
                import pandas as pd

                merged_existing_df = pd.concat(self.conformation_all_df_list)
            self.global_pocket_seq_data = GlobalPocketSeqDataset(
                self,
                istrain=istrain,
                existing_df=merged_existing_df,
            )

        self.train_dataset, self.valid_dataset, self.test_dataset = self._compose_splits()
        self.datasets = (self.train_dataset, self.valid_dataset, self.test_dataset)

    def get_embedding_parts(
        self,
        reaction_csv_path,
        substrate_csv_path,
        sequence_csv_path,
        product_csv_path,
    ):
        from .data_part.global_sequence_part import (
            get_or_build_sequence_embedding,
        )
        from .data_part.pocket_part import get_or_build_pocket_embedding
        from .data_part.substrate_product_part import (
            get_or_build_product_embedding,
            get_or_build_substrate_embedding,
        )
        from .data_part.ts_reaction_part import get_or_build_ts_reaction_embedding
        from .data_part.rxn_fp_part import get_or_build_reaction_fp_embedding

        substrate_em = None
        product_em = None
        sequence_em = None
        pocket_em = None
        reaction_fp_em = None
        reaction_em_use = None

        if self.load_substrate_embedding and substrate_csv_path:
            substrate_em = get_or_build_substrate_embedding(self, substrate_csv_path)

        if self.load_product_embedding and product_csv_path:
            product_em = get_or_build_product_embedding(self, product_csv_path)

        if self.load_sequence_embedding and sequence_csv_path:
            sequence_em = get_or_build_sequence_embedding(
                self,
                sequence_csv_path,
                output_name=self.sequence_embedding_output,
            )

        if self.load_pocket_embedding and sequence_csv_path:
            pocket_em = get_or_build_pocket_embedding(
                self,
                sequence_csv_path,
                output_name="pocket_gvp_graph",
            )

        if self.load_reaction_fp_embedding:
            rxn_fp_emb_path = self.fingerprint.get("rxn_fp_emb_path", None)
            rxn_fp_mask_path = self.fingerprint.get("rxn_fp_mask_path", None)
            if rxn_fp_emb_path:
                reaction_fp_em = get_or_build_reaction_fp_embedding(
                    self,
                    rxn_fp_emb_path,
                    rxn_fp_mask_path,
                )

        if self.load_reaction_embedding and reaction_csv_path:
            reaction_em_use = get_or_build_ts_reaction_embedding(self, reaction_csv_path)

        return reaction_em_use, substrate_em, sequence_em, pocket_em, product_em, reaction_fp_em

    def _compose_splits(self):
        if self.dataset_mode == "complex_only":
            return self.complex_only_data.datasets
        if self.dataset_mode == "global_pocket_seq":
            return self.global_pocket_seq_data.datasets

        struct_train, struct_valid, struct_test = self.complex_only_data.datasets
        seq_train, seq_valid, seq_test = self.global_pocket_seq_data.datasets
        return (
            ConcatDatasetSplit(struct_train, seq_train),
            ConcatDatasetSplit(struct_valid, seq_valid),
            ConcatDatasetSplit(struct_test, seq_test),
        )

    


class ConcatDatasetSplit(Dataset):
    def __init__(self, dsA, dsB):
        self.dsA = dsA if dsA is not None else []
        self.dsB = dsB if dsB is not None else []
        self.lenA = len(self.dsA)
        self.lenB = len(self.dsB)

    def __len__(self):
        return self.lenA + self.lenB

    def __getitem__(self, idx):
        if idx < self.lenA:
            return self.dsA[idx]
        return self.dsB[idx - self.lenA]
