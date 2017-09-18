# ADPPortfolioSelection

## Introduction

This is the Python project corresponding to my Master Thesis "Stochastic Dyamic Programming applied to Portfolio Selection problem".
My report can be found on [my ResearchGate profile](https://www.researchgate.net/publication/317958018_Stochastic_Dynamic_Programming_for_Portfolio_Selection_Problem_applied_to_CAC40 "Report on ResearchGate").
This project is also in the continuity of [another project](https://www.researchgate.net/publication/317951607_Scenario-Based_Portfolio_Optimization), which is a study of different risk measures of portfolio management, based on Scenarios Generation.

## Installation

This project uses Python version 3. However, you *have to use python3.5 maximum*, because you need to install `gurobipy`, which is the Python API of an optimisation library called [Gurobi](http://www.gurobi.com/ "Gurobi Website").

I strongly encourage you to install the project inside a `virtualenv` environment:

    virtualenv -p python3.5 env
    source env/bin/activate

Then, the main dependencies can be installed via pip:

    pip install -r requirement.txt
   
You can download Gurobi [on their website](http://www.gurobi.com/registration/download-reg "Gurobi download page") and install it.
Then, go into the directory (for instance, `/Library/gurobi702/mac64` for gurobi v7.02 for Mac 64-bits), and launch (while you're still in the `python3.5` virtual environment):

    python setup.py install

You should be setup to launch the project!
