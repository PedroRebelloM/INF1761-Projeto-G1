from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State
  from pipeline import Pipeline
  from transform import Transform
  from appearance import Appearance
  from shape import Shape

class Node:
  """A scene graph node: holds an optional Pipeline/Transform, this node's
  own Appearances and Shapes, and child Nodes. Pipeline, Transform, and
  Appearances (materials, texture sets) are all inherited by descendants
  that don't set their own - only Shapes are strictly local to a Node."""

  def __init__ (self, pipeline: Pipeline | None = None, trf: Transform | None = None, apps: list[Appearance] | None = None, shps: list[Shape] | None = None, nodes: list[Node] | None = None) -> None:
    """Builds a node from already-constructed parts; `nodes` (if given) are
    attached via add_node."""
    self.parent: Node | None = None
    self.pipeline = pipeline
    self.trf = trf
    self.apps: list[Appearance] = apps or []
    self.shps: list[Shape] = shps or []
    self.nodes: list[Node] = []
    if nodes:
      for n in nodes:
        self.add_node(n)

  def set_pipeline (self, pipeline: Pipeline) -> None:
    """Sets the Pipeline this node loads on render; descendants that don't
    set their own inherit it from an ancestor."""
    self.pipeline = pipeline

  def get_pipeline (self) -> Pipeline | None:
    """Returns the Pipeline set directly on this node, or None if it only
    inherits one from an ancestor."""
    return self.pipeline

  def set_transform (self, trf: Transform | None) -> None:
    """Sets the local Transform composed into this node's model matrix, or
    clears it (None) back to identity; descendants without their own
    Transform get identity here, not an ancestor's."""
    self.trf = trf

  def add_appearance (self, app: Appearance) -> None:
    """Appends `app` to this node's appearance list. Loaded in order
    (unloaded in reverse) before this node's Shapes are drawn, so later
    appearances can override earlier ones."""
    self.apps.append(app)

  def add_shape (self, shp: Shape) -> None:
    """Appends `shp` to this node's shape list; drawn in order under
    whichever Pipeline is active when this node renders."""
    self.shps.append(shp)

  def add_node (self, node: Node) -> None:
    """Appends `node` as a child and sets this node as its parent.
    Raises if `node` already has a parent (use remove_node on the
    current parent first - a Node can only ever be a child of one
    parent, or it'd end up in two `nodes` lists at once, rendered
    twice, while get_parent()/get_model_matrix() only ever see the
    most recent one) or if `node` is `self` or one of `self`'s own
    ancestors (which would make the tree an infinite loop for
    render()/get_model_matrix() to walk)."""
    if node.parent is not None:
      raise ValueError("Node already has a parent - remove it from its current parent first")
    ancestor: Node | None = self
    while ancestor is not None:
      if ancestor is node:
        raise ValueError("Node is self or an ancestor of self - would create a cycle")
      ancestor = ancestor.parent
    self.nodes.append(node)
    node.set_parent(self)

  def remove_node (self, node: Node) -> None:
    """Detaches `node` from this Node's children and clears its parent
    pointer, so it (or a subtree containing it) can be added elsewhere.
    Raises if `node` isn't currently a child of this Node."""
    if node not in self.nodes:
      raise ValueError("node is not a child of this Node")
    self.nodes.remove(node)
    node.parent = None

  def set_parent (self, parent: Node) -> None:
    """Sets this node's parent pointer, used by get_model_matrix to walk up
    the ancestor chain. Called automatically by add_node, not meant to be
    called directly."""
    self.parent = parent

  def get_parent (self) -> Node | None:
    """Returns this node's parent, or None if it's the root (or not yet
    attached to a tree)."""
    return self.parent

  def get_matrix (self) -> gm.Mat4:
    """Returns this node's own local transform matrix, or the identity if
    no Transform is set."""
    if self.trf:
      return self.trf.get_matrix()
    else:
      return gm.mat4(1.0)

  def get_model_matrix (self) -> gm.Mat4:
    """Returns the accumulated model matrix: this node's local matrix
    composed with every ancestor's, up to the root."""
    mat = self.get_matrix()
    node = self.get_parent()
    while node:
      mat = node.get_matrix() @ mat
      node = node.get_parent()
    return mat

  def render (self, st: State) -> None:
    """Recursively renders this node's subtree: loads its own
    Pipeline/Transform/Appearances, draws its own Shapes, renders every
    child Node, then unloads everything in reverse order."""
    # load
    if self.pipeline:
      self.pipeline.load(st)
    if self.trf:
      self.trf.load(st)
    for app in self.apps:
      app.load(st)
    # draw
    if len(self.shps) > 0:
      # whichever Pipeline is current is already bound on the render pass:
      # Pipeline.load binds it on the way in, and a nested Pipeline's
      # unload rebinds the enclosing one on the way out
      st.load_matrices()
      for shp in self.shps:
        shp.draw(st)
      st.unload_matrices()
    for node in self.nodes:
      node.render(st)
    # unload in reverse order
    for app in reversed(self.apps):
      app.unload(st)
    if self.trf:
      self.trf.unload(st)
    if self.pipeline:
      self.pipeline.unload(st)
