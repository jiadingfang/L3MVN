import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import quaternion

from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.habitat.objectgoal_env21 import ObjectGoal_Env21
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu

from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping
import torch


class Sem_Exp_Env_Agent(ObjectGoal_Env21):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.device = args.device
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.red_sem_pred = load_rednet(
            self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        )
        self.red_sem_pred.eval()
        # self.red_sem_pred.to(self.device)


        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        self.replan_count = 0
        self.collision_n = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

        self.fail_case = {}
        self.fail_case['collision'] = 0
        self.fail_case['success'] = 0
        self.fail_case['detection'] = 0
        self.fail_case['exploration'] = 0

        self.eve_angle = 0

    def reset(self):
        args = self.args

        self.replan_count = 0
        self.collision_n = 0

        obs, info = super().reset()
        obs = self._preprocess_obs(obs)

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None

        self.eve_angle = 0
        self.eve_angle_old = 0

        info['eve_angle'] = self.eve_angle


        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # s_time = time.time()

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), self.fail_case, False, self.info

        # Reset reward if new long-term goal
        llm_sp_frontier_compatible = -1
        if planner_inputs["new_goal"]:
            goal = planner_inputs['goal']
            map_target = planner_inputs["map_target"]

            if np.sum(goal == 1) == 1 and self.args.task_config == "tasks/objectnav_gibson.yaml":
                frontier_loc = np.where(goal == 1)
                self.info["g_reward"] = self.get_llm_distance(planner_inputs["map_target"], frontier_loc)
            
            # if np.sum(goal == 1) == 1 and len(list(set(map_target.ravel()))) >= (4 + 1): # 0 is background, 1,2,3,4 are frontiers 
            #     frontier_loc = np.where(goal == 1)
            #     self.info["g_reward"] = self.get_llm_distance(planner_inputs, frontier_loc)

            # if np.sum(goal == 1) == 1 and len(list(set(map_target.ravel()))) >= (4 + 1): # 0 is background, 1,2,3,4 are frontiers
            #     frontier_loc = np.where(goal == 1)
            #     llm_sp_frontier_compatible = self.check_llm_shortest_frontier_compatible(planner_inputs, frontier_loc)

            self.collision_map = np.zeros(self.visited.shape)
            self.info['clear_flag'] = 0

        action = self._plan(planner_inputs)

        # c_time = time.time()
        # ss_time = c_time - s_time
        # print('plan map: %.3f秒'%ss_time) 0.19

        if self.collision_n > 20 or self.replan_count > 26:
            self.info['clear_flag'] = 1
            self.collision_n = 0

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        if action >= 0:

            # act
            action = {'action': action}
            obs, rew, done, info = super().step(action)
            # print('super obs shape: ', obs.shape) # [5, 480, 640]

            if done and self.info['success'] == 0:
                if self.info['time'] >= self.args.max_episode_length - 1:
                    self.fail_case['exploration'] += 1
                elif self.replan_count > 26:
                    self.fail_case['collision'] += 1
                else:
                    self.fail_case['detection'] += 1

            if done and self.info['success'] == 1:
                self.fail_case['success'] += 1

            # preprocess obs
            obs = self._preprocess_obs(obs)
            # print('obs shape after preprocess: ', obs.shape) # [20, 120, 160]
            self.last_action = action['action']
            self.obs = obs
            self.info = info
            info['eve_angle'] = self.eve_angle
            info['llm_sp_frontier_compatible'] = llm_sp_frontier_compatible

            # e_time = time.time()
            # ss_time = e_time - c_time
            # print('act map: %.3f秒'%ss_time) 0.23

            # info['g_reward'] += rew

            return obs, self.fail_case, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), self.fail_case, False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        if replan:
            self.replan_count += 1
            print("replan_count: ", self.replan_count)
        else:
            self.replan_count = 0

        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) or self.replan_count > 26:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x > map_pred.shape[0]: eve_start_x = map_pred.shape[0] 
            if eve_start_y > map_pred.shape[0]: eve_start_y = map_pred.shape[0] 
            if eve_start_x < 0: eve_start_x = 0 
            if eve_start_y < 0: eve_start_y = 0 
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:,:,4:5].squeeze()
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1

            sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
            sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
            sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
            sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
            sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1

        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth, use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy()
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return red_semantic_pred, semantic_pred

    def round_view_of_frontier(self, frontier_pos, frontier_rot, n=16, save=True, name=None):
        def display_obs(obs, name=""):
            img = obs["rgb"]
            depth = obs["depth"]
            semantic = obs["semantic"]

            arr = [img, depth, semantic]
            titles = ["rgb", "depth", "semantic"]
            plt.figure(figsize=(12, 8))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 3, i + 1)
                ax.axis("off")
                ax.set_title(titles[i])
                plt.imshow(data)

            # plt.show()
            plt.savefig("test_viz/goal_viz_{}.png".format(name))
            plt.close()

        list_sim_obs = []
        for i in range(n):
            d_rot_rad = i / n * np.pi * 2
            d_rot_quat = quaternion.from_rotation_vector([0, d_rot_rad, 0])
            rot_quat = d_rot_quat * frontier_rot
            # print('frontier_pos: ', frontier_pos)
            # print('rot_quat: ', rot_quat)
            frontier_obs = self._env.sim.get_observations_at(frontier_pos, rot_quat)
            list_sim_obs.append(frontier_obs)
            if save:
                if name is None:
                    display_obs(frontier_obs, "pos_{}_{}".format(frontier_pos, i))
                else:
                    display_obs(frontier_obs, "{}_{}".format(name, i))
        return list_sim_obs
    
    def map_coords_to_sim_coords(self, origin, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position.
        """
        # use planner inputs to get origin from lmb to convert local_map coords to global_map coords and then convert to sim coords
        agent_state = self._env.sim.get_agent_state(0)
        agent_position = agent_state.position
        # min_x, min_y = self.map_origin[0] / 100.0, self.map_origin[1] / 100.0
        x, y = coords
        cont_x = x / 20. + origin[0] / 20.
        cont_y = (480 - y) / 20. + origin[1] / 20.
        # agent_position[0] = cont_y
        # agent_position[2] = cont_x
        agent_position[0] = -0.455 * cont_x - 0.891 * cont_y + 39.487
        agent_position[2] = -0.891 * cont_x + 0.455 * cont_y + 15.362

        return agent_position

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/good_0.1/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        local_w = inputs['map_pred'].shape[0]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        map_edge = inputs['map_edge']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        edge_mask = map_edge == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        if np.sum(goal) == 1:
            f_pos = np.argwhere(goal == 1)

            goal_fmb = skimage.draw.circle_perimeter(f_pos[0][0], f_pos[0][1], int(local_w/4-2))
            goal_fmb[0][goal_fmb[0] > local_w-1] = local_w-1
            goal_fmb[1][goal_fmb[1] > local_w-1] = local_w-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)

        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1) * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            map_pose = pos
            agent_state = self._env.sim.get_agent_state(0)
            agent_position = agent_state.position
            agent_rot = agent_state.rotation
            
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-map_pose-{}-sim_pose-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep, map_pose, agent_position)
            cv2.imwrite(fn, self.vis_image)

            print('full pose (start_x, start_y, start_o): ', start_x, start_y, start_o)
            print('origins (gx1, gy1): ', gx1, gy1)
            print('local map boundary: [gx1, gx2, gy1, gy2]: ', gx1, gx2, gy1, gy2)
            print('map pose: ', map_pose)
            print('agent postion: ', agent_position)
            agent_x, agent_y, _ = map_pose
            agent_map_coords = [agent_x, agent_y]
            agent_origin = [gy1, gx1]

            # agent_cont_x = (agent_x + gy1) / 20. 
            # agent_cont_y = (480 - agent_y + gx1) / 20. 
            # agent_calc_x = -0.455 * agent_cont_x - 0.891 * agent_cont_y + 39.487
            # agent_calc_y = -0.891 * agent_cont_x + 0.455 * agent_cont_y + 15.362
            # # print('calculated agent position: ', calc_x, calc_y)
            # agent_position[0] = agent_calc_x
            # agent_position[2] = agent_calc_y
            
            calc_agent_position = self.map_coords_to_sim_coords(agent_origin, agent_map_coords)
            print('calc agent position: ', calc_agent_position)
            self.round_view_of_frontier(calc_agent_position, agent_rot, n=8, name='timestep_{}_calc_agent_view_{}'.format(self.timestep, agent_position))

            # # test print
            # for i in range(11):
            #     for j in range(11):
            #         agent_position[0] = i - 5
            #         agent_position[2] = j - 5
            #         self.round_view_of_frontier(agent_position, agent_rot, n=1, name='test_agent_view_{}'.format(agent_position))

            if np.sum(goal) == 1:
                print('f_pos: ', f_pos)
                goal_x, goal_y = f_pos[0]
                goal_map_coords = [goal_x, goal_y]
                goal_origin = [gy1, gx1]

                # goal_cont_x = (goal_x + gy1) / 20. 
                # goal_cont_y = (480 - goal_y + gx1) / 20. 
                # goal_calc_x = -0.455 * goal_cont_x - 0.891 * goal_cont_y + 39.487
                # goal_calc_y = -0.891 * goal_cont_x + 0.455 * goal_cont_y + 15.362
                # agent_position[0] = goal_calc_x
                # agent_position[2] = goal_calc_y
                
                calc_goal_position = self.map_coords_to_sim_coords(origin=goal_origin, coords=goal_map_coords)
                print('goal position: ', calc_goal_position)
                self.round_view_of_frontier(calc_goal_position, agent_rot, n=8, name='timestep_{}_goal_position_view_{}'.format(self.timestep, agent_position))

                # test print
                n_points = 5
                for i in range(n_points+1):
                    mid_x = agent_x + (goal_x - agent_x) / n_points * i
                    mid_y = agent_y + (goal_y - agent_y) / n_points * i
                    mid_map_coords = [mid_x, mid_y]
                    mid_origin = [gy1, gx1]
                    calc_mid_position = self.map_coords_to_sim_coords(origin=mid_origin, coords=mid_map_coords)
                    obs = self.round_view_of_frontier(calc_mid_position, agent_rot, n=1, save=False)
                    obs_rgb = obs[0]["rgb"]

                    # print('sem_map shape: ', sem_map.shape) # (480, 480)
                    mid_fmb = skimage.draw.circle_perimeter(int(mid_x), int(mid_y), 3)
                    mid_fmb[0][mid_fmb[0] > local_w-1] = local_w-1
                    mid_fmb[1][mid_fmb[1] > local_w-1] = local_w-1
                    mid_fmb[0][mid_fmb[0] < 0] = 0
                    mid_fmb[1][mid_fmb[1] < 0] = 0
                    # goal_fmb[goal_fmb < 0] =0
                    edge_mask[mid_fmb[0], mid_fmb[1]] = 1
                    sem_map[edge_mask] = 3
                    color_pal = [int(x * 255.) for x in color_palette]
                    test_sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
                    test_sem_map_vis.putpalette(color_pal)
                    test_sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
                    test_sem_map_vis = test_sem_map_vis.convert("RGB")
                    test_sem_map_vis = np.flipud(test_sem_map_vis)
                    test_sem_map_vis = test_sem_map_vis[:, :, [2, 1, 0]]
                    test_sem_map_vis = cv2.resize(test_sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)

                    test_image = vu.init_vis_image(self.goal_name, self.legend)
                    test_image[50:530, 15:655] = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)
                    test_image[50:530, 670:1150] = test_sem_map_vis

                    cv2.drawContours(test_image, [agent_arrow], 0, color, -1)
                    fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-mid_pose-{}-sim_pose-{}-idx-{}.png'.format(
                        dump_dir, self.rank, self.episode_no,
                        self.rank, self.episode_no, self.timestep, map_pose, agent_position, i)
                    cv2.imwrite(fn, test_image)
