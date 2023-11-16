### Introduction

This is the code implementation (Pytorch) to solve the issues on the ASR of CTRL. Here is the official code of CTRL (https://github.com/meet-cjli/CTRL/tree/master), and next, we will demonstrate how we make the result closer to our reported performance (a more practical scenario) in our paper.

We focus on the official scenario of CTRL, where the victim uses **ResNet18** as a backbone and trains it on **CIFAR-10** using **SimCLR**. This is also the scenario adopted by **ASSET[1] (reviewer wKXh refers to this paper**).

[1] Pan, Minzhou, et al. "ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms." Usenix Security (2023).

### Preprocessing

Before running, we make the following adjustment on SimCLR code of CTRL in "main_train.py" (**official code which can report similar ACC (80+%) and ASR (80+%) in CTRL and [1]**) and "main_train_my.py" (**our code (adjusted a little from the official code) which can report similar ACC (90+%) and ASR (20%)**):

(1) We turn the temperature of SimCLR from 0.5 (official setting) to 0.2 (most popular setting).
(2) We select the official settings of magnitude (50 on data poisoning) and (100 on ASR performance evaluation).

### Then, let's begin！

1. Run the following code to verify **the offical code of CTRL.** If you don't have enough time to verify the performance, you can check **log/original_setting.log** on the running result directly.
```
python main_train.py --data_path [your CIFAR-10 dir place]
```

Here we simply illustrate the several lines of **log/original_setting_simclr.log**.

> [602-epoch] time:29.118 | knn acc: 79.256 | back acc: 80.170 | loss:7.983 | cl_loss:7.983
>
> [1000-epoch] time:28.330 | knn acc: 83.069 | back acc: 81.950 | loss:7.178 | cl_loss:7.178

2. Run the following code to **our improved code of CTRL.** If you don't have enough time to verify the performance, you can check **log/change_train_dataset_testset.log** on the running result directly.

```
python main_train_my.py --data_path [your CIFAR-10 dir place]
```

Here we simply illustrate the several lines of **log/change_train_dataset_testset.log**.

> [600-epoch] time:20.704 | knn acc: 88.250 | back acc: 15.690 | loss:7.912 | cl_loss:7.912
>
> [1000-epoch] time:21.229 | knn acc: 90.500 | back acc: 18.260 | loss:7.142 | cl_loss:7.142

### What have caused the difference between ours result and theirs?

In brief, the official code of CTRL adopts unusual training implementations of SimCLR, leading to a low ACC (80% in CTRL and 85% in [1]; the ACC of SimCLR over CIFAR-10 can easily reach 90%!) and a high ASR (once the victim use normalize augmentation during the performance validation, the ASR will drop heavily). Most importantly, if the victim adjusts the above issues in CTRL's code, he will gain a **natural ACC** (90%, just like our log change_train_dataset_testset.log). Our settings are more practical, as nowadays, more trainers prefer a higher performance on ACC (our 90+% instead of 80+%).


(1) Their training implementation on input datasets are unusual (which leads to the low ACC of CTRL and [1]). Besides, using this configuration, the training doesn't reach convergence yet, according to the loss curve (provided in their official code: https://github.com/meet-cjli/CTRL/tree/master) in "img/training.jpg". We list their code as follows (you can find it in "methods/base.py"):

```
for i, (images, __, _) in enumerate(train_loader):  #frequency backdoor has been injected
    #print(i)
    model.train()
    images = images.cuda(self.args.gpu, non_blocking=True)

    #data
    v1 = train_transform(images)
    v2 = train_transform(images)
```

where the train_transform is (you can find it in "loaders/diffaugment.py", line 351):

```
train_transform = nn.Sequential(aug.RandomResizedCrop(size = (args.size, args.size), scale=(0.2, 1.0)),
                                                 aug.RandomHorizontalFlip(),
                                                 RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                                 aug.RandomGrayscale(p=0.2),
                                                 normalize)
```

Such a way of data augmentation will render the whole data batch in v1 to adopt the same augmentation with the training step (and so is v2). In a typical case, **the trainer will insert the data augmentation within the train_loader**. Just like what we do (you can find our train_loader setting in "loaders/diffaugment.py," line 152):

And our adjusted training code is shown as follows (you can find it in "methods/base.py," line 332):


```
for i, ((v1, v2), _) in enumerate(train_loader):  #frequency backdoor has been injected
    #print(i)
    model.train()
    # images = images.cuda(self.args.gpu, non_blocking=True)

    #data
    # v1 = train_transform(images)
    # v2 = train_transform(images)

```

(2) Whether the memory_set and test_set utilize the ``normalize'' will significantly affect the performance of ASR. **We argue this is because of the change of decision boundary.** In the official code of CTRL, whether the train set utilizes "normalize" in data augmentations or not, the memory set and test set don't employ "normalize" (see code in "base.py," line 428). Such a validation strategy is unusual, too, as it will lead to the degradation of ACC **For a victim, they will select to add the normalize for a higher ACC, so we argue that our attacking scenarios are more practical than theirs (CTRL and [1])**.


Our adjustment on this part is let the utilization of normalize return to normal:

this is the CTRL's official set on memory, test test:

```
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor, train_index), batch_size=self.args.batch_size, sampler=train_sampler, shuffle= (train_sampler is None), drop_last=True)
        test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor, test_index), batch_size=self.args.eval_batch_size, shuffle=False, drop_last=True)
        test_pos_loader = DataLoader(TensorDataset(x_test_pos_tensor, y_test_pos_tensor, test_index), batch_size=self.args.eval_batch_size, shuffle=False)
        memory_loader = DataLoader(TensorDataset(x_memory_tensor, y_memory_tensor, memory_index), batch_size=self.args.eval_batch_size, shuffle=False)
```

where these tensors (such as x_train_tensor) is unaugmented verison of CIFAR-10 input.

This is ours, we insert the normalize into the memory, test test:

```
train_transform_ori = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(rf"transform used: {train_transform_ori}")
        print(r"batch size is ", self.args.batch_size)
        train_transform_by = lambda x: (train_transform_ori(x), train_transform_ori(x))
        train_loader = DataLoader(TensorsDataset_Offline(x_train_tensor, y_train_tensor, train_transform_by),
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        test_loader = DataLoader(TensorsDataset_Offline(x_test_tensor, y_test_tensor, test_transform),
                                 batch_size=self.args.eval_batch_size, shuffle=False)
        test_pos_loader = DataLoader(TensorsDataset_Offline(x_test_pos_tensor, y_test_pos_tensor, test_transform),
                                     batch_size=self.args.eval_batch_size, shuffle=False)
        memory_loader = DataLoader(TensorsDataset_Offline(x_memory_tensor, y_memory_tensor, test_transform),
                                   batch_size=self.args.eval_batch_size, shuffle=False)
```

[1] Pan, Minzhou, et al. "ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms." Usenix Security (2023).