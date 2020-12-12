# Pytorch-Lightning-Templates

Set of training loops for solving common deep learning problems using PyTorch Lightning workflow. Each subproject contains examples of the data format that is expected. The project aims to make prototyping models as easy as possible while benefiting from all the "training tricks" that every deep learning practitioner should use.

For every subproject the training loop is defined in [train.py](base_scripts/base_train.py) file. Its structure is similar in each case: 

```python
checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=2,
        verbose=True,
        monitor="val/loss",
        mode="min",
    )

    experiment_name = ...
    PROJECT_NAME = ...

    logger = WandbLogger(name=experiment_name, project=PROJECT_NAME)

    # And then actual training
    pl.seed_everything(42)
    trainer = Trainer(
        max_epochs=40,
        logger=logger,
        gpus=1,
        # precision=16,
        deterministic=True,
        accumulate_grad_batches=2,
        callbacks=[EarlyStopping(monitor="val/loss")],
        # resume_from_checkpoint = 'my_checkpoint.ckpt'
    )

    trainer.fit(model, dm)
```

Defaultly we assume the user is logged into Wandb. To do so you need to run `!wandb login your_personal_hash_generated_by_wandb` in the terminal


## Avalible templates

- [Image classification](image_classification/)
- [Multi label image classification](image_classification/)
- [Object detection](image_classification/)
- [Image Segmentation](image_classification/)


## Authors
* [Piotr Mazurek](https://github.com/tugot17)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
