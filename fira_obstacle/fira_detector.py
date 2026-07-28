import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class FIRASegmentationResult:
    blue_walls: List[np.ndarray]
    yellow_holes_bounds: List[np.ndarray]
    red_gates: List[np.ndarray]

@dataclass
class FIRAObstacles:
    boundaries: List[np.ndarray]
    holes: List[np.ndarray]
    walls: List[np.ndarray]
    gates: List[Tuple[np.ndarray, List[np.ndarray]]] # (Red gate top contour, list of supporting blue pillars)
    heading_target: Optional[Tuple[int, int]] = None


class FIRADetector:
    def __init__(self):
        # HSV color ranges for segmentation
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([140, 255, 255])
        
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])
        
    def _segment_colors(self, image: np.ndarray) -> FIRASegmentationResult:
        """
        Step 1: Segment the frame into Blue (Walls), Yellow (Holes/Bounds), and Red (Gates).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        kernel = np.ones((5,5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        
        blue_contours = self._extract_valid_contours(mask_blue, min_area=500)
        yellow_contours = self._extract_valid_contours(mask_yellow, min_area=500)
        red_contours = self._extract_valid_contours(mask_red, min_area=500)
        
        return FIRASegmentationResult(blue_walls=blue_contours, yellow_holes_bounds=yellow_contours, red_gates=red_contours)
        
    def _extract_valid_contours(self, mask: np.ndarray, min_area: int) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, FIRAObstacles]:
        """
        Step 2: Structurally classify the segmented colors into exact obstacle types.
        """
        annotated_image = image.copy()
        seg_result = self._segment_colors(image)
        
        boundaries = []
        holes = []
        walls = []
        gates = []
        
        # 1. Classify Yellow: Boundaries (long lines touching image edges) vs Holes (compact, floating)
        h, w = image.shape[:2]
        for cnt in seg_result.yellow_holes_bounds:
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Boundary lines typically stretch to the edges of the camera view
            touches_edge = (x < 10) or (x + bw > w - 10) or (y + bh > h - 10) or (y < 10)
            
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue
            aspect_ratio = max(rw, rh) / min(rw, rh)
            area = cv2.contourArea(cnt)
            
            if touches_edge and (aspect_ratio > 2.5 or area > 3000):
                boundaries.append(cnt)
                cv2.drawContours(annotated_image, [cnt], -1, (0, 255, 255), 2)
                cv2.putText(annotated_image, "BOUNDARY", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                holes.append(cnt)
                cv2.drawContours(annotated_image, [cnt], -1, (0, 200, 200), 3)
                cv2.putText(annotated_image, "HOLE", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                
        # 2. Classify Blue and Red: Gates vs Walls
        unassigned_blue = list(seg_result.blue_walls)
        
        for red_cnt in seg_result.red_gates:
            rx, ry, rw, rh = cv2.boundingRect(red_cnt)
            gate_pillars = []
            remaining_blue = []
            
            for blue_cnt in unassigned_blue:
                bx, by, bw, bh = cv2.boundingRect(blue_cnt)
                
                # Check if blue pillar is horizontally aligned with the red gate
                overlap_x = max(0, min(rx + rw, bx + bw) - max(rx, bx))
                
                # Check if blue pillar is vertically below the red gate (allowing some tolerance)
                is_below = by > ry - 50 
                
                if overlap_x > 0 and is_below:
                    gate_pillars.append(blue_cnt)
                else:
                    remaining_blue.append(blue_cnt)
                    
            unassigned_blue = remaining_blue
            gates.append((red_cnt, gate_pillars))
            
            # Annotate Gate
            cv2.drawContours(annotated_image, [red_cnt], -1, (0, 0, 255), 3)
            cv2.putText(annotated_image, "GATE_TOP", (rx, max(15, ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for pillar in gate_pillars:
                cv2.drawContours(annotated_image, [pillar], -1, (255, 150, 50), 3)
                px, py, pw, ph = cv2.boundingRect(pillar)
                cv2.putText(annotated_image, "PILLAR", (px, max(15, py-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)
                
        # 3. Remaining Blue are Walls
        for blue_cnt in unassigned_blue:
            walls.append(blue_cnt)
            cv2.drawContours(annotated_image, [blue_cnt], -1, (255, 0, 0), 3)
            bx, by, bw, bh = cv2.boundingRect(blue_cnt)
            cv2.putText(annotated_image, "WALL", (bx, max(15, by-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        # 4. Step 3: Pathfinding - Calculate safe heading
        target_pt = None
        
        # Priority 1: Aim for the middle of a gate
        if len(gates) > 0:
            red_cnt, pillars = gates[0]
            if len(pillars) >= 2:
                # Find the center between the first two pillars
                px1, py1, pw1, ph1 = cv2.boundingRect(pillars[0])
                px2, py2, pw2, ph2 = cv2.boundingRect(pillars[1])
                target_x = (px1 + pw1//2 + px2 + pw2//2) // 2
                target_y = max(py1, py2) + min(ph1, ph2) // 2
                target_pt = (target_x, target_y)
        
        # Priority 2: Find the largest horizontal gap avoiding walls and holes
        if target_pt is None:
            impassable_boxes = [cv2.boundingRect(w_cnt) for w_cnt in walls] + [cv2.boundingRect(h_cnt) for h_cnt in holes]
            scan_y = h // 2
            occupied = np.zeros(w, dtype=bool)
            
            for bx, by, bw, bh in impassable_boxes:
                # If obstacle overlaps with our scan line
                if by <= scan_y <= by + bh:
                    margin = 50 # Safe distance from obstacle in pixels
                    start_x = max(0, bx - margin)
                    end_x = min(w, bx + bw + margin)
                    occupied[start_x:end_x] = True
            
            max_gap_start = 0
            max_gap_len = 0
            current_start = -1
            
            for i in range(w):
                if not occupied[i]:
                    if current_start == -1:
                        current_start = i
                else:
                    if current_start != -1:
                        gap_len = i - current_start
                        if gap_len > max_gap_len:
                            max_gap_len = gap_len
                            max_gap_start = current_start
                        current_start = -1
                        
            if current_start != -1:
                gap_len = w - current_start
                if gap_len > max_gap_len:
                    max_gap_len = gap_len
                    max_gap_start = current_start
                    
            if max_gap_len > 0:
                target_x = max_gap_start + max_gap_len // 2
                target_pt = (target_x, scan_y)
            else:
                target_pt = (w // 2, scan_y) # Fallback straight ahead
                
        # Draw heading
        start_pt = (w // 2, h)
        cv2.arrowedLine(annotated_image, start_pt, target_pt, (0, 255, 0), 4, tipLength=0.1)
        
        result = FIRAObstacles(
            boundaries=boundaries,
            holes=holes,
            walls=walls,
            gates=gates,
            heading_target=target_pt
        )
        
        return annotated_image, result
