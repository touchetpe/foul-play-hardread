> **HardRead fork note:** this branch is HardRead's fork of Foul Play by pmariglia. Results HardRead reports were measured at the evaluated commit f061e63 (public id 175e591), before this fix.
>
> At 175e591 the bot crashes in games against a species missing from its usage table, such as Roaring Moon, which joined OU on 30 October. The error is `AttributeError: 'list' object has no attribute 'get'` in `SmogonSets.move_usage_rates`. This branch adds a small guard: a species with no statistics gets no moves to sample. In our seeded checks, sampling for species in the table didn't change. The guard stops the crash but adds no usage data for Roaring Moon. HardRead's review has its own guard (`tools/review/fpenv.py` in [touchetpe/hardread](https://github.com/touchetpe/hardread)).

# Foul Play ![umbreon](https://play.pokemonshowdown.com/sprites/xyani/umbreon.gif)
A Pokémon battle-bot that can play battles on [Pokemon Showdown](https://pokemonshowdown.com/).

Foul Play can play single battles in all generations
though currently dynamax and z-moves are not supported.

![badge](https://github.com/pmariglia/foul-play/actions/workflows/ci.yml/badge.svg)

## Python version
Requires Python 3.11+.

## Getting Started

### Configuration

Command-line arguments are used to configure Foul Play

use `python run.py --help` to see all options.

### Running Locally

**1. Clone**

Clone the repository with `git clone https://github.com/pmariglia/foul-play.git`

**2. Install Requirements**

Install the requirements with `pip install -r requirements.txt`.

Note: Requires Rust to be installed on your machine to build the engine.

**4. Run**

Run with `python run.py`

Here is a minimal example that plays a gen9randombattle on Pokemon Showdown:
```bash
python run.py \
--websocket-uri wss://sim3.psim.us/showdown/websocket \
--ps-username 'My Username' \
--ps-password sekret \
--bot-mode search_ladder \
--pokemon-format gen9randombattle
```

### Running with Docker

**1. Clone the repository**

`git clone https://github.com/pmariglia/foul-play.git`

**2. Build the Docker image**

Use the `Makefile` to build a Docker image
```shell
make docker
```

or for a specific generation:
```shell
make docker GEN=gen4
```

**3. Run the Docker Image**
```bash
docker run --rm --network host foul-play:latest \
--websocket-uri wss://sim3.psim.us/showdown/websocket \
--ps-username 'My Username' \
--ps-password sekret \
--bot-mode search_ladder \
--pokemon-format gen9randombattle
```

## Engine

This project uses [poke-engine](https://github.com/pmariglia/poke-engine) to search through battles.
See [the engine docs](https://poke-engine.readthedocs.io/en/latest/) for more information.

The engine must be built from source if installing locally so you must have rust installed on your machine.

### Re-Installing the Engine

It is common to want to re-install the engine for different generations of Pokémon.

`pip` will used cached .whl artifacts when installing packages
and cannot detect the `--config-settings` flag that was used to build the engine.

The following command will ensure that the engine is re-installed properly:
```shell
pip uninstall -y poke-engine && pip install -v --force-reinstall --no-cache-dir poke-engine --config-settings="build-args=--features poke-engine/<GENERATION> --no-default-features"
```

Or using the Makefile:
```shell
make poke_engine GEN=<generation>
```

For example, to re-install the engine for generation 4:
```shell
make poke_engine GEN=gen4
```
