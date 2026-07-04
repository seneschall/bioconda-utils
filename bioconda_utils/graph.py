"""
Construction and Manipulation of Package/Recipe Graphs
"""

import logging

from collections import defaultdict
from fnmatch import fnmatch
from itertools import chain
from pathlib import Path
from typing import (
    Any,
)
from collections.abc import Iterable, Iterator, Sequence

import networkx as nx
import rattler_build as rb

from bioconda_utils.recipe import Recipe
from bioconda_utils.skiplist import Skiplist

from . import utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def build(
    recipes: Iterable[utils.RecipePath],
    config: dict[str, Any],
    blacklist: Skiplist | None = None,
    restrict: bool = True,
) -> tuple[nx.DiGraph[str], defaultdict[str, set[utils.RecipePath]]]:
    """
    Returns the DAG of recipe paths and a dictionary that maps package names to
    lists of recipe paths to all defined versions of the package.  defined
    versions.

    Parameters
    ----------
    recipes : iterable
        An iterable of recipe paths, typically obtained via `get_recipes()`

    blacklist : set
        Package names to skip

    restrict : bool
        If True, then dependencies will be included in the DAG only if they are
        themselves in `recipes`. Otherwise, include all dependencies of
        `recipes`.

    Returns
    -------
    dag : nx.DiGraph
        Directed graph of packages -- nodes are package names; edges are
        dependencies (both run and build dependencies)

    name2recipe : dict
        Dictionary mapping package names to recipe paths. These recipe path
        values are lists and contain paths to all defined versions.
    """
    logger.info("Generating DAG")
    recipes: list[utils.RecipePath] = list(recipes)
    # TODO (rb): fix load meta fast so it returns the correct type
    # we have to replace this with a function that returns both
    meta_rattler_data: list[utils.MetaOrRattler] = list(
        utils.parallel_iter(utils.load_meta_and_recipe_fast, recipes, "Loading Recipes")
    )

    # name2recipe is meta.yaml's / recipe.yaml's package:name mapped to the recipe path.
    #
    # A name should map to exactly one recipe. It is possible for multiple
    # names to map to the same recipe, if the package name somehow depends on
    # the environment.
    name2recipe: defaultdict[str, set[utils.RecipePath]] = defaultdict(set)

    for rendered_recipe in meta_rattler_data:
        name: str = ""
        if rendered_recipe.meta is not None:
            # i.e. buildsystem is conda-build
            name: str = rendered_recipe.meta["package"]["name"]
        elif rendered_recipe.rattler is not None:
            # i.e. build system is rattler-build
            name: str = rendered_recipe.rattler[0].recipe.package.name
        else:
            raise ValueError(
                f"No meta or rattler-recipe found for: {rendered_recipe.path.path.as_posix()}"
            )
        if blacklist is None or not blacklist.is_skiplisted(rendered_recipe.path.path):
            name2recipe[name].update([rendered_recipe.path])

    def get_deps(meta: dict[str, Any], sec: str) -> list[str]:
        reqs = meta.get("requirements")
        if not reqs:
            return []
        deps = reqs.get(sec)
        if not deps:
            return []
        return [dep.split()[0] for dep in deps if dep]

    def get_inner_deps(dependencies: Iterable[str]) -> Iterator[str]:
        dependencies = list(dependencies)
        for dep in dependencies:
            if dep in name2recipe or not restrict:
                yield dep

    # TODO (rb): is it more efficient to merge this with the loop above?
    dag: nx.DiGraph[str] = nx.DiGraph()
    dag.add_nodes_from(name2recipe.keys())
    for rendered_recipe in meta_rattler_data:
        if rendered_recipe.meta is not None:
            # i.e. buildsystem is conda-build
            meta: dict[str, Any] = rendered_recipe.meta
            name: str = meta["package"]["name"]
            dag.add_edges_from(
                (dep, name)
                for dep in set(
                    chain(
                        get_inner_deps(get_deps(meta, "build")),
                        get_inner_deps(get_deps(meta, "host")),
                        get_inner_deps(get_deps(meta, "run")),
                    )
                )
            )
        elif rendered_recipe.rattler is not None:
            # i.e. build system is rattler-build
            rendered_variants: list[rb.RenderedVariant] = rendered_recipe.rattler
            name: str = rendered_variants[0].recipe.package.name

            for v in rendered_variants:
                dag.add_edges_from(
                    (dep, name)
                    for dep in set(
                        chain(
                            get_inner_deps(v.recipe.requirements.build),
                            get_inner_deps(v.recipe.requirements.host),
                            get_inner_deps(v.recipe.requirements.run),
                        )
                    )
                )
        else:
            raise ValueError(
                f"No meta or rattler-recipe found for: {rendered_recipe.path.path.as_posix()}"
            )

    return dag, name2recipe


def build_from_recipes(recipes: Iterable[Recipe]) -> nx.DiGraph:
    logger.info("Building Recipe DAG")

    package2recipes = {}
    recipe_list = []
    for recipe in recipes:
        for package in recipe.package_names:
            package2recipes.setdefault(package, set()).add(recipe)
        recipe_list.append(recipe)

    dag = nx.DiGraph()
    dag.add_nodes_from(recipe for recipe in recipe_list)
    dag.add_edges_from(
        (recipe2, recipe)
        for recipe in recipe_list
        for dep in recipe.get_deps()
        for recipe2 in package2recipes.get(dep, [])
    )

    logger.info(
        "Building Recipe DAG: done (%i nodes, %i edges)",
        len(dag),
        len(dag.edges()),
    )
    return dag


def filter_recipe_dag(
    dag: nx.DiGraph, include: Sequence[str], exclude: Sequence[str]
) -> nx.DiGraph:
    """Reduces **dag** to packages in **names** and their requirements"""
    nodes = set()
    for recipe in dag:
        if (
            recipe not in nodes
            and any(fnmatch(recipe.reldir, p) for p in include)
            and not any(fnmatch(recipe.reldir, p) for p in exclude)
        ):
            nodes.add(recipe)
            nodes |= nx.ancestors(dag, recipe)
    return nx.subgraph(dag, nodes)


def filter(dag: nx.DiGraph, packages: Iterable[str]) -> nx.DiGraph:
    nodes = set()
    for package in packages:
        if package in nodes:
            continue  # already got all ancestors
        nodes.add(package)
        try:
            nodes |= nx.ancestors(dag, package)
        except nx.exception.NetworkXError:
            if package not in nx.nodes(dag):
                logger.error("Can't find %s in dag", package)
            else:
                raise

    return nx.subgraph(dag, nodes)


def is_leaf(dag: nx.DiGraph, pkg_name: str) -> bool:
    return dag.out_degree(pkg_name) == 0
