from modules import PaperModel, BaselineModel
import nibabel as nib
import torch

inp = "/home/andrey/Documents/comp3710/project/keras_slices_data/keras_slices_seg_test/seg_040_week_0_slice_0.nii.gz"
img = nib.load(inp)
model = PaperModel(5)
model.train()
input = torch.tensor(img.get_fdata(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
out = model(input)
# print(out.shape)

model = BaselineModel(5)
model.train()
out = model(input)
# print(out.shape)