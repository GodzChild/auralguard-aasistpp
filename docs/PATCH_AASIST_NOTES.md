# AASIST patch notes

The current official AASIST model already returns:

```python
return last_hidden, output
```

So this project can wrap it directly.

If your local copy only returns `output`, modify the final lines of `models/AASIST.py` from:

```python
output = self.out_layer(last_hidden)
return output
```

to:

```python
output = self.out_layer(last_hidden)
return last_hidden, output
```

Then `AuralGuardAASISTPP` can use `last_hidden` for the extra heads.
