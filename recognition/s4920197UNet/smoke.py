from modules import PaperModel, BaselineModel
import nibabel as nib
import torch
import matplotlib.pyplot as plt

inp = "/home/andrey/Documents/comp3710/project/keras_slices_data/keras_slices_train/case_027_week_6_slice_44.nii.gz"
img = nib.load(inp).get_fdata()
model = PaperModel()
model.train()
input = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
out = model(input)
print(out.shape)

model = BaselineModel()
model.train()
out = model(input)
print(out)

plt.imshow(img)
plt.show()

