# ML-Template
Template repository for machine learning research projects.


## Getting Started

### Requirements

After cloning this template repository, you should install
the default requirements (you can modify them to suit your needs first) using
the command
```commandline
pip install -r requirements.txt
```

### Experiments

This repository is organized around the concept of **experiments**. An
experiment defines a process that can be executed, which may depend
on input (hyper-) parameters called **configuration**. A particular execution
of an experiment is called a **run**.

Every experiment in a project has an **experiment name**, which is an
identifier string that is unique within the project. To create an experiment,
simply create a Python package under the `experiments` package. The experiment
name is the same as the name of the experiment package in `experiments`.

To define your experiment, add a Python module `<experiment name>.py` to the
experiment pacakge, where `<experiment name>` is the experiment name. Then, 
create a class in this package that derives from `experiments.Experiment`. An
experiment named `demo` is already provided as a demonstration of this 
structure.

### Running experiments

To run an experiment, use the `run_experiment` command with the name of the
experiment. For example, to run the `demo` experiment, use
```commandline
python -m run_experiment demo
```

### Configuration

Experiments may be run with different input parameters, which are called
experiment **configuration**. Configuration must be a Python `dict` that can
be serialized into JSON or YAML. As such, configuration is specified using JSON
or YAML files.

The `run_experiment` command searches for configuration files in the following
locations:

- `config.(yaml|json)`, which contains project global configuration
- `experiments/<experiment_name>/config.(yaml|json)`, which contains experiment
   default configuration
- `experiments/<experiment_name>/<config_name>.config.(yaml|json)`, which
   contains optional experiment configuration

Note that none of these configuration files are necessary; running without them
will simply add no options to the configuration, and will not result in an 
error.

The global `config.(yaml|json)` configuration will always be used, with values
being overridden by the experiment default configuration and then by any other 
configuration files specified. Options are overridden recursively, so if the
global configuration is
```python
global_config = {
    'a': {
        'b': 1,
        'c': 2
    }
}
```
and the experiment default configuration is
```python
experiment_config = {
    'a': {
        'b': 2
    }
}
```
then the final configuration will be given by
```python
config = {
    'a': {
        'b': 2,
        'c': 2
    }
}
```
_not_
```python
config = {
    'a': {
        'b': 2
    }
}
```
as would be the case if options were overridden only at the top level.

To specify additional override configuration files, provide their names
(with or without file extension; path to the file relative to the experiment
directory will also work) as extra arguments to the `run_experiment`
command. For example, the `demo` experiment provides an extra configuration file
named `other.yaml`, so we can run `demo` using this override by using
```commandline
python -m run_experiment demo other
```
Any number of additional override files may be specified, and they will override
the global and experiment default configuration in the order they are given.

### Configuration groups

It is often necessary to run the same experiment with many slightly different
configurations. This can be achieved in one command with `run_experiment` by
using **configuration groups**. A configuration group is a set of configuration
files grouped together in one directory. If an experiment is run with a 
configuration group, then it will be run once for each configuration file in the
group, with the options from that file overriding any other configuration
specified in `run_experiment`. 

To run an experiment with a configuration group, use the optional `--group`
argument of `run_experiment` to specify the location of the configuration group
directory relative to the experiment directory. For example, the `demo`
experiment has a configuration group called `test_group`. To run the `demo`
experiment for this configuration group we use
```commandline
python -m run_experiment demo --group test_group
```
Note that the option `group_id` will be added to the configuration of each
run in an experiment group, with the value being the name of the corresponding 
configuration file (without file extension).

To use the configuration options in the `other.yaml` configuration file when
running the `test_group` configuration group of the `demo` experiment, we
provide the name `other` as an additional argument, as before, noting that
additional override configuration files must be specified _before_ `--group`,
as follows
```commandline
python -m run_experiment demo other --group test_group
```

### Side experiments

It may be desirable to have several main experiments, each of which has several
corresponding smaller, related experiments. This can be implemented using
**side experiments**, which are experiments related to regular experiments but
defined within the related experiment's package. The related experiment is 
called the **main experiment**

A side experiment's name is given by the main experiment's name followed by a
forward slash (or the system file separator character) and then an identifier
for the side experiment. To define a side  experiment, create a subclass of 
`experiments.Experiment` in a module in the main experiment package that has 
the same name as the identifier of the side experiment. For example, the `demo`
experiment has a side experiment named `demo/side`, which is implemented in the
file `experiments/demo/side.py`.

Side experiments can be run in exactly the same way that normal experiments
are run. For example, to run the `demo/side` experiment, use
```commandline
python -m run_experiment demo/side
```
Configuration is specified to side experiments using the same arguments and
conventions that apply to main experiments; however, after the experiment
default configuration is applied and before any other configuration is applied,
any configuration specified in the optional file
`<identifier>.config.(json|yaml)` will be applied, where `<identifier>` is the
local identifier of the side experiment.

### Sub-experiments

It is also possible to create an experiment pacakge within another experiment
package. An experiment defined in this way is called a **sub-experiment**. The
name of a sub-experiment is the path to its package directory relative to the
`experiments` directory, much like side experiments are named. Following this
convention it is possible to nest sub-experiments as deeply as one likes. 

If a side experiment and sub-experiment of the same local identifier exist within
the same main experiment package, creating a naming ambiguity, then the side
experiment will be given precedence by`run_experiment`. Side experiments can be
placed within sub-experiments, and in this case the name of the side experiment
is still its path relative to the `experiments` directory.

The `demo` experiment has a sub-experiment named `demo/sub`, which we can run
using
```commandline
python -m run_experiment demo/sub
```
The `demo/sub` sub-experiment has its own side experiment named `demo/sub/side`,
which we can run with
```commandline
python -m run_experiment demo/sub/side
```
Note that, unlike side experiments, the parent experiment default configuration
is _not_ used when running a sub-experiment. If this behavior is desired, then
the parent configuration file must be specified manually as an optional 
override, like so
```commandline
python -m run_experiment demo/sub ../config
```