from core import decomp, grid

import collections

class Solver:

  grid = None
  target_size = 0
  rects = []
  start_cell_full = {}
  
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

    self.target_size = dec_size
      
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
    res = 1e6
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      if inds[direc] != ind:
        continue
      else:
        if inds[res_dir] < res:
          res = inds[res_dir]
    return res
    
  def __max_start_inds_at(self, direc, ind, res_dir):
    res = 0
    for r in self.rects:
      inds = r.get_start_cell().int_bounds()
      if inds[direc] != ind:
        continue
      else:
        if inds[res_dir] > res:
          res = inds[res_dir]
    return res
    
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
      # Check the four borders, can figure out corner cells through starts and ends
      left_edge1 = rect1.get_start_cell().int_bounds()[0]
      bot_edge1 = rect1.get_start_cell().int_bounds()[1]
      right_edge1 = rect1.get_end_cell().int_bounds()[0]
      top_edge1 = rect1.get_end_cell().int_bounds()[1]
      
      left_edge2 = rect2.get_start_cell().int_bounds()[0]
      bot_edge2 = rect2.get_start_cell().int_bounds()[1]
      right_edge2 = rect2.get_end_cell().int_bounds()[0]
      top_edge2 = rect2.get_end_cell().int_bounds()[1]   

      return ((top_edge1 == bot_edge2 - 1 and rect1.length(0) == rect2.length(0))
             or (bot_edge1 == top_edge2 + 1 and rect1.length(0) == rect2.length(0))
             or (right_edge1 == left_edge2 - 1 and rect1.length(1) == rect2.length(1))
             or (left_edge1 == right_edge2 + 1 and rect1.length(1) == rect2.length(1)))

    if not __border_share(rect1, rect2):
      print("Rectangles do not share a border, aborting merge")
      return None
      
    start1 = rect1.get_start_cell()
    start2 = rect2.get_start_cell()
    
    # Find the lower indexed rectangle to initialize the new merged one
    
    if start1.int_bounds()[0] < start2.int_bounds()[0] or start1.int_bounds()[1] < start2.int_bounds()[1]:
      return decomp.Decomp(self.grid, start1, rect2.get_end_cell())
    else: 
      return decomp.Decomp(self.grid, start2, rect1.get_end_cell())
 
  def run_solver(self):
    merge_dir = -1
    merge_count = 0
    
    while (len(self.start_cell_full.values()) > self.target_size):
      merge_dir = (merge_dir + 1) % 3
      print("Merge direction is ", merge_dir)
      min_inds = self.__min_start_inds()
      max_inds = self.__max_start_inds()
      
      search_dir1 = (merge_dir + 1) % 3
      search_dir2 = (merge_dir + 2) % 3
      failed_dirs = 0
      merged = False
      
      for i in range(min_inds[merge_dir], max_inds[merge_dir] + 1):
        min_inds1 = self.__min_start_inds_at(merge_dir, i, search_dir1)
        min_inds2 = self.__min_start_inds_at(merge_dir, i, search_dir2)
        max_inds1 = self.__max_start_inds_at(merge_dir, i, search_dir1)
        max_inds2 = self.__max_start_inds_at(merge_dir, i, search_dir2)
        
        for j in range(min_inds1, max_inds1 + 1):
          for k in range(min_inds2, max_inds2 + 1):
            base_rect_inds = self.__min_start_inds()
            # TODO: The above line only works if we have uniform geometry
            base_rect_inds[merge_dir] = i
            base_rect = self.__rect_lookup(base_rect_inds)
            if base_rect == None:
              print("Failed merge: no rectangle with ", i, j, k)
              continue

            rect_to_merge_inds = [0, 0, 0]
            rect_to_merge_inds[merge_dir] = i
            rect_to_merge_inds[search_dir1] = j
            rect_to_merge_inds[search_dir2] = k
            rect_to_merge = self.__rect_lookup(rect_to_merge_inds)
            if rect_to_merge == None:
              print("Failed merge: couldn't find rect with indices ", i, j, k)
              continue
              
            merged_rect = self.merge_rect(base_rect, rect_to_merge)
            if merged_rect != None:
              del self.start_cell_full[tuple(rect_to_merge_inds)]
              del self.start_cell_full[tuple(base_rect_inds)]
              self.start_cell_full.update({tuple(merged_rect.get_start_cell().int_bounds()) : merged_rect})

              merged = True
              print("Merge successful: rect goes from ", merged_rect.get_start_cell().int_bounds(), " to ", merged_rect.get_end_cell().int_bounds())
              merge_count = merge_count + 1
              failed_dirs = 0
              if len(self.start_cell_full.values()) <= self.target_size:
                return list(self.start_cell_full.values())
            else:
              print("Merge failed: rect at", base_rect.get_start_cell().int_bounds(), "couldn't merge with", rect_to_merge.get_start_cell().int_bounds())

      if not merged:
        failed_dirs = failed_dirs + 1
        if failed_dirs == 3:
          print("No more legal merges found, returning decomp of size ", len(self.start_cell_full.values()), " instead.")
          break
      merged = False
              
    return list(self.start_cell_full.values())