import torch

data_root_path = "toy_example"
pocket_ligand_emb_path = "toy_example"
align_model_path = 'checkpoint/align_modal'
pred_checkpoint_path = "checkpoint/save_model"
save_model_path = "checkpoint/save_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")