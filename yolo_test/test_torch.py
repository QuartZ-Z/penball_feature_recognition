import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
gpu_list = [i for i in range(torch.cuda.device_count())]
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())