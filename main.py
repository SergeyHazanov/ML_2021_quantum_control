from Trainer import Trainer

energy_gap, runtime, dt, skips = 1, 20, 0.01, 5
trainer = Trainer(energy_gap, runtime, dt, skips)
trainer.train()
