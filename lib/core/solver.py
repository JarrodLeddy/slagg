from core import decomp, grid

import collections

"""
To implement: to prevent stalling and loops...

1) On initializaiton, for each geo cell, count its neighboring geo cells (corners don't count) on that layer
   - If 0, something went very wrong; abort with error message
   - If 1, mark LEDGE
   - If 2 and not all three share an index in one direction, mark VERTEX
   - If 2 and the neighbors are in a line, mark EDGE
   - If 3, mark EDGE
   - If 4, mark INTERIOR
   
   Use enum to help with this
   
2) Create a set (or dictionary) of all vertex points for reference in the solver code

3) Make sure the merged boolean is visible in the while-loop scope

4) Instead of taking the base_rect at __min_start_inds(), select a rectangle that uses a vertex cell instead
   4a) Figure out a way to merge "backwards" as well i.e. a higher indexed rectangle into a lower one

5) If merged is false after an iteration through i...
   - Increase the failed_dir count by 1
   - If failed_dir = 3, assume no more work is to be done with that vertex and kick it out of the dictionary
   - Return partial decomp if the dictionary is empty at that point
   - Else, reset failed_dir to 0 and go back to the top
   
Make this happen for Monday or at least show the framework for Leddy
"""

class Solver:

  grid = None
  target_size = 0
  rects = []
  start_cell_full = {}
  vert_rects = []
  
  def __init__(self, grid, dec_size):
    self.grid = grid
    for i in range(grid.get_dims()[0]):
      for j in range(grid.get_dims()[1]):
        for k in range(grid.get_dims()[2]):
          if grid.get_cell(i, j, k).has_geometry():
            self.rects.append(decomp.Decomp(grid, grid.get_cell(i, j, k), grid.get_cell(i, j, k)))
            
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      self.start_cell_full[tuple(inds)] = r

    for r in self.rects:
      # Figure cell status in relation to geometry
      neighbor_count = 0
      is_edge = False
      base_inds = r.get_start_cell().int_bounds()
      if self.__rect_lookup([base_inds[0] + 1, base_inds[1], base_inds[2]]) is not None:
        neighbor_count = neighbor_count + 1
      if self.__rect_lookup([base_inds[0] - 1, base_inds[1], base_inds[2]]) is not None:
        neighbor_count = neighbor_count + 1
        if self.__rect_lookup([base_inds[0] + 1, base_inds[1], base_inds[2]]) is not None:
          is_edge = True
      if self.__rect_lookup([base_inds[0], base_inds[1] + 1, base_inds[2]]) is not None:
        neighbor_count = neighbor_count + 1
      if self.__rect_lookup([base_inds[0], base_inds[1] - 1, base_inds[2]]) is not None:
        neighbor_count = neighbor_count + 1
        if self.__rect_lookup([base_inds[0], base_inds[1] + 1, base_inds[2]]) is not None:
          is_edge = True
        
      if neighbor_count == 1:
        r.set_state(decomp.RectStartState.LEDGE)
        self.__rect_lookup(base_inds).set_state(decomp.RectStartState.LEDGE)
      elif neighbor_count == 3:
        r.set_state(decomp.RectStartState.EDGE)
        self.__rect_lookup(base_inds).set_state(decomp.RectStartState.EDGE)
      elif neighbor_count == 4:
        r.set_state(decomp.RectStartState.INTERIOR)
        self.__rect_lookup(base_inds).set_state(decomp.RectStartState.INTERIOR)
      elif neighbor_count == 2:
        if is_edge:
          r.set_state(decomp.RectStartState.EDGE)
          self.__rect_lookup(base_inds).set_state(decomp.RectStartState.EDGE)
        else:
          r.set_state(decomp.RectStartState.VERTEX)
          self.__rect_lookup(base_inds).set_state(decomp.RectStartState.VERTEX)
      else:
        # Make an exception class here to explain what went wrong, we're only here if neighbor_count is 0 or more than 4
        r.set_state(decomp.RectStartState.NONE)
        self.__rect_lookup(base_inds).set_state(decomp.RectStartState.NONE)

    self.target_size = dec_size
    self.vert_rects = [r for r in self.rects if r.get_state() == decomp.RectStartState.VERTEX]
      
  def __check_rect_geometry(self):
  # Check each rectangle to see if it's full of geometry
  # If there is any blank space, flag it
    res = []
    for r in rects:
      if r.has_vacancies:
        res.append(False)
      else:
        res.append(True)
    return res
    
  def __rect_lookup(self, inds):
  # Given a set of indices, check if any rectangle in the current decomp
  # has a starting cell corresponding to them
    if tuple(inds) in self.start_cell_full:
      return self.start_cell_full[tuple(inds)]
    return None

  def __min_start_inds(self):
    # Retrun lowest valid index in each direction
    res = [1e6, 1e6, 1e6]
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      for i in range(len(inds)):
        if inds[i] < res[i]:
          res[i] = inds[i]
    return res
    
  def __max_start_inds(self):
    # Return highest valid index in each direction
    res = [0, 0, 0]
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      for i in range(len(inds)):
        if inds[i] > res[i]:
          res[i] = inds[i]
    return res
    
  def __min_start_inds_at(self, direc, ind, res_dir):
    res = -1
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      if inds[direc] != ind:
        continue
      else:
        if inds[res_dir] < res or res == -1:
          res = inds[res_dir]
    return int(res)
    
  def __max_start_inds_at(self, direc, ind, res_dir):
    res = -1
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      if inds[direc] != ind:
        continue
      else:
        if inds[res_dir] > res or res == -1:
          res = inds[res_dir]
    return int(res)
    
  def setup_solver(self):
  # Place all the rectangles on the lowest section of the geometry
  # Solver will shift and resize them later
    for cell in grid.get_cells():
      if cell.has_geometry():
        for r in self.rects:
          r.shift(cell.int_bounds())
          
  def cell_owner(self, cell):
  # If a cell is covered by a rectangle, identify who's covering it
  # If multiple, assume the lower indexed one
    target_inds = cell.int_bounds()
    for i in range(rects.size):
      if rects[i].contains_inds(target_inds):
        return rects[i]
      else:
        return None
        
  def split_rect(self, rect):
    start_inds = rect.get_start_cell.int_bounds()
    end_inds = rect.get_end_cell.int_bounds()
    
    # Compute a midpoint cell
    mid_ind_x = (end_inds[0] - start_inds[0]) / 2
    mid_ind_y = (end_inds[1] - start_inds[1]) / 2
    
    new_rects = []
    new_rects.append(Decomp(grid.get_cell(start_inds[0], start_inds[1]), grid.get_cell(mid_ind_x, mid_ind_y)))
    new_rects.append(Decomp(grid.get_cell(mid_ind_x, mid_ind_y), grid.get_cell(end_inds[0], end_inds[1])))
    return new_rects
    
  def merge_rect(self, rect1, rect2):
    # First assert there's a shared border with the two rectangles
    
    def __border_share(rect1, rect2):
      # Check the six borders, can figure out corner cells through starts and ends
      left_edge1 = rect1.get_start_cell().int_bounds()[0]
      bot_edge1 = rect1.get_start_cell().int_bounds()[1]
      front_edge1 = rect1.get_start_cell().int_bounds()[2]
      right_edge1 = rect1.get_end_cell().int_bounds()[0]
      top_edge1 = rect1.get_end_cell().int_bounds()[1]
      back_edge1 = rect1.get_end_cell().int_bounds()[2]
      
      left_edge2 = rect2.get_start_cell().int_bounds()[0]
      bot_edge2 = rect2.get_start_cell().int_bounds()[1]
      front_edge2 = rect2.get_start_cell().int_bounds()[2]
      right_edge2 = rect2.get_end_cell().int_bounds()[0]
      top_edge2 = rect2.get_end_cell().int_bounds()[1] 
      back_edge2 = rect2.get_end_cell().int_bounds()[2]

      def __length_check(rect1, rect2, ind):
        return rect1.length(ind) == rect2.length(ind)  
        
      return ((top_edge1 == bot_edge2 - 1 and __length_check(rect1, rect2, 0) and __length_check(rect1, rect2, 2) and       left_edge1 == left_edge2 and right_edge1 == right_edge2)
             or (bot_edge1 == top_edge2 + 1 and __length_check(rect1, rect2, 0) and __length_check(rect1, rect2, 2) and left_edge1 == left_edge2 and right_edge1 == right_edge2)
             or (right_edge1 == left_edge2 - 1 and __length_check(rect1, rect2, 1) and __length_check(rect1, rect2, 2) and top_edge1 == top_edge2 and bot_edge1 == bot_edge2)
             or (left_edge1 == right_edge2 + 1 and __length_check(rect1, rect2, 1) and __length_check(rect1, rect2, 2) and top_edge1 == top_edge2 and bot_edge1 == bot_edge2)
             or (back_edge1 == front_edge2 - 1 and __length_check(rect1, rect2, 0) and __length_check(rect1, rect2, 1))
             or (front_edge1 == back_edge2 + 1 and __length_check(rect1, rect2, 0) and __length_check(rect1, rect2, 1)))

    if not __border_share(rect1, rect2):
      print("Rectangles do not share a border, aborting merge")
      return None
      
    start1 = rect1.get_start_cell()
    start2 = rect2.get_start_cell()
    
    # Find the lower indexed rectangle to initialize the new merged one
    
    if start1.int_bounds()[0] < start2.int_bounds()[0] or start1.int_bounds()[1] < start2.int_bounds()[1] or start1.int_bounds()[2] < start2.int_bounds()[2]:
      return decomp.Decomp(self.grid, start1, rect2.get_end_cell())
    else: 
      return decomp.Decomp(self.grid, start2, rect1.get_end_cell())
 
  def run_solver(self):
    merge_dir = -1
    merge_count = 0
    dir_changes = 0
    first_pass = True
    backward = True
    merged_last = False
    
    last_j = -1
    last_k = -1
    full_pass = False
    
    def reset_stored_inds():
      last_j = -1
      last_k = -1
    
    while (len(self.start_cell_full.values()) > self.target_size):
      min_inds = self.__min_start_inds()
      max_inds = self.__max_start_inds()
      
      # Only change the loop direction if no merges were found on the last run
      if not merged_last:
        backward = False if backward else True
    
      if not full_pass and not backward:
        merge_dir = (merge_dir + 1) % 3
      full_pass = True
      search_dir1 = (merge_dir + 1) % 3
      search_dir2 = (merge_dir + 2) % 3
      failed_dirs = 0
      # Loop over vertex points?
      # Eventually...for now we can do a greedy approach and just make sure we don't
      # merge two cells on opposite ends of the geometry. Just enforce boundaries
      
      i_range = None
      j_range = None
      k_range = None
      # For negative looping
      if not backward:
        i_range = range(min_inds[merge_dir], max_inds[merge_dir] + 1)
      else:
        i_range = range(max_inds[merge_dir], min_inds[merge_dir] - 1, -1)

      for i in i_range:
        min_inds1 = self.__min_start_inds_at(merge_dir, i, search_dir1)
        min_inds2 = self.__min_start_inds_at(merge_dir, i, search_dir2)
        max_inds1 = self.__max_start_inds_at(merge_dir, i, search_dir1)
        max_inds2 = self.__max_start_inds_at(merge_dir, i, search_dir2)
        merged = False
        
        if not backward:        
          j_range = range(min_inds1, max_inds1 + 1)
          k_range = range(min_inds2, max_inds2 + 1)
        else:
          j_range = range(max_inds1, min_inds1 - 1, -1)
          k_range = range(max_inds2, min_inds2 - 1, -1)
        
        for j in j_range:
          if merged:
            break
          for k in k_range:
            if merged:
              break

            rect_to_merge_inds = [0, 0, 0]
            rect_to_merge_inds[merge_dir] = i
            rect_to_merge_inds[search_dir1] = j
            rect_to_merge_inds[search_dir2] = k
            rect_to_merge = self.__rect_lookup(rect_to_merge_inds)
            if rect_to_merge is None:
              # print("Failed merge: couldn't find rect with indices ", i, j, k)
              full_pass = False
              continue
                        
            base_rect_inds = self.__min_start_inds()
            if len(self.vert_rects) > 0:
              base_rect_inds = self.vert_rects[0].get_start_cell().int_bounds()
            # TODO: The above line only works if we have uniform geometry
            base_rect_inds[merge_dir] = i
            
            # Fix for constant layer merges
            base_rect_inds[2] = rect_to_merge.get_start_cell().int_bounds()[2]
            # if last_j >= 0:
              # base_rect_inds[search_dir1] = last_j
            # if last_k >= 0:
              # base_rect_inds[search_dir2]= last_k
            base_rect = self.__rect_lookup(base_rect_inds)
            if base_rect is None:
              # print("Failed merge: no rectangle with ", i, j, k)
              # del(self.vert_rects[0])
              full_pass = False
              continue
              
            # Also make sure these indices actually border each other
            merged_rect = self.merge_rect(base_rect, rect_to_merge)
            if merged_rect is not None:
              del self.start_cell_full[tuple(rect_to_merge_inds)]
              del self.start_cell_full[tuple(base_rect_inds)]
              self.start_cell_full.update({tuple(merged_rect.get_start_cell().int_bounds()) : merged_rect})

              merged = True
              last_j = merged_rect.get_start_cell().int_bounds()[search_dir1]
              last_k = merged_rect.get_start_cell().int_bounds()[search_dir2]
              print("Merge successful: rect goes from ", merged_rect.get_start_cell().int_bounds(), " to ", merged_rect.get_end_cell().int_bounds())
              merge_count = merge_count + 1
              failed_dirs = 0
              if len(self.start_cell_full.values()) <= self.target_size:
                return list(self.start_cell_full.values())
            else:
              full_pass = False

      if not merged:
        failed_dirs = failed_dirs + 1
        if failed_dirs == 3:
          print("No more legal merges found, returning decomp of size ", len(self.start_cell_full.values()), " instead.")
          # del(self.vert_rects[0])
          break

      # Then check if the larger merged rectangles can be brought in themselves
      # This first pass will only compare similarly indexed rectangles on the merge coordinate
            

    return list(self.start_cell_full.values())