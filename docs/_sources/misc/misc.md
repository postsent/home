# Misc
# References

Colab - https://colab.research.google.com/notebooks/pro.ipynb#scrollTo=23TOba33L4qf

# Jupyter Notebook

```py
%matplotlib inline 
from IPython.core.interactiveshell import InteractiveShell
get_ipython().ast_node_interactivity = 'all'

def p(t=''):
    print('-------'+t+'-------')
    print()
```

# Colab



## Faster GPUs

With Colab Pro you have priority access to our fastest GPUs and with Pro+ even more so. For example, you may get a T4 or P100 GPU at times when most users of standard Colab receive a slower K80 GPU. You can see what GPU you've been assigned at any time by executing the following cell.

```py
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info) 
```

## More memory

With Colab Pro you have the option to access high-memory VMs when they are available, and with Pro+ even more so. To set your notebook preference to use a high-memory runtime, select the Runtime > 'Change runtime type' menu, and then select High-RAM in the Runtime shape dropdown.

```python
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
```

## Longer runtimes

While Colab Pro subscribers still have limits, these will be roughly **twice** the limits for non-subscribers, with even more stability for Pro+.