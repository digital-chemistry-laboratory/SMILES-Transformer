.. (other content above) ...

Getting Started
===============
+++++++++
Installation
+++++++++
To install the package, just run:

.. code-block:: bash

    git clone https://github.com/your-username/smiles_transformer.git
    cd smiles_transformer
    pip install .

+++++++++
Quick Start
+++++++++
Here is a quick example of how to use the library:

- Create a `WandB <https://wandb.ai/>`_ account and login locally.
- Modify the config file found at ``configurations/config.yaml`` to suit your use-case. The documentation for the config file can be found here: :doc:`configurations`
- Once done, run the main file to train your model:

.. code-block:: python

  python -m main.py "path/to/configuration/file"

Alternatively, run a cross-validation run:

.. code-block:: python

  python -m launch_cv.py "path/to/configuration/file"

.. (other content) ...

.. toctree::
   :hidden:
   :caption: Shortcuts
   :maxdepth: 1

   Home <self>
   configurations

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   modules
