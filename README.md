# Theseus Lab

An isolated terminal prototype for testing the Ship of Theseus thesis:
behavior can remain stable while the mechanism changes, splits, or disappears.

This is an in-memory logic sandbox. It does not load model weights, call a
provider, read assessment data, or make production claims. The prototype is
deliberately small so the state transitions remain inspectable.

Run the full demonstration:

```sh
python3 theseus_lab.py --demo
```

Run an interactive session:

```sh
python3 theseus_lab.py
```

Commands are `show`, `preserve OLD`, `replace OLD NEW`, `split OLD NEW_A
NEW_B`, `probe`, `reset`, `help`, and `quit`.

The durable question and current verdict live in [NOTES.md](NOTES.md).
