import os
import json
import bz2
import gzip
import _pickle as cPickle
import gym
import cv2
import numpy as np
import quaternion
import skimage.morphology
import habitat
from PIL import Image
import copy

from envs.utils.fmm_planner import FMMPlanner
from constants import category_to_id, mp3d_category_id
import envs.utils.pose as pu
from constants import color_palette
import agents.utils.visualization as vu


import matplotlib.pyplot as plt

coco_categories = [0, 3, 2, 4, 5, 1]

class ObjectGoal_Env21(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        # self.scene = self._env.sim.semantic_annotations()

        fileName = 'data/matterport_category_mappings.tsv'

        text = ''
        lines = []
        items = []
        self.hm3d_semantic_mapping={}

        with open(fileName, 'r') as f:
            text = f.read()
        lines = text.split('\n')

        for l in lines:
            items.append(l.split('    '))

        for i in items:
            if len(i) > 3:
                self.hm3d_semantic_mapping[i[2]] = i[-1]
        # hm3d_semantic_mapping is a map between actual object name to category name

        # print()

        # for obj in self.scene.objects:
        #     if obj is not None:
        #         print(
        #             f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
        #             f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        #         )

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        # new_scene = self.episode_no % args.num_train_episodes == 0


        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        # if new_scene:
        obs = super().reset()
        start_height = 0
        self.scene = self._env.sim.semantic_annotations()
        # start_height = self._env.current_episode.start_position[1]
        # goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        # floor_height = []
        # floor_size = []
        # for obj in self.scene.objects:
        #     if obj.category.name() in self.hm3d_semantic_mapping and \
        #         self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #         floor_height.append(abs(obj.aabb.center[1] - start_height))
        #         floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        
        # if not args.eval:
        #     while all(h > 0.1 or s < 12 for (h,s) in zip(floor_height, floor_size)) or abs(start_height-goal_height) > 1.2:
        #         obs = super().reset()

        #         self.scene = self._env.sim.semantic_annotations()
        #         start_height = self._env.current_episode.start_position[1]
        #         goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        #         floor_height = []
        #         floor_size = []
        #         for obj in self.scene.objects:
        #             if obj.category.name() in self.hm3d_semantic_mapping and \
        #                 self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #                 floor_height.append(abs(obj.aabb.center[1] - start_height))
        #                 floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        self.prev_distance = self._env.get_metrics()["distance_to_goal"]
        self.starting_distance = self._env.get_metrics()["distance_to_goal"]


        # print("obs: ,", obs)
        # print("obs shape: ,", obs.shape)
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        semantic = self._preprocess_semantic(obs["semantic"])
        # print("rgb shape: ,", rgb.shape)
        # print("depth shape: ,", depth.shape)
        # print("semantic shape: ,", semantic.shape)

        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = coco_categories[obs['objectgoal'][0]]
        self.info['goal_name'] = category_to_id[obs['objectgoal'][0]]
        self.info['agent_state'] = self._env.sim.get_agent_state(0)

        self.goal_name = category_to_id[obs['objectgoal'][0]]

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        semantic = self._preprocess_semantic(obs["semantic"])
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        # print('rgb shape: ', rgb.shape) # [480, 640, 3]
        # print('depth shape: ', depth.shape) # [480, 640, 1]
        # print('semantic shape: ', semantic.shape) # [480, 640, 1]
        # print('state shape: ', state.shape) # [5, 480, 640]

        self.timestep += 1
        self.info['time'] = self.timestep
        self.info['agent_state'] = self._env.sim.get_agent_state(0)

        return state, rew, done, self.info

    def _preprocess_semantic(self, semantic):
        # print("*********semantic type: ", type(semantic))
        se = list(set(semantic.ravel()))
        # print(se) # []
        for i in range(len(se)):
            if self.scene.objects[se[i]].category.name() in self.hm3d_semantic_mapping:
                hm3d_category_name = self.hm3d_semantic_mapping[self.scene.objects[se[i]].category.name()]
            else:
                hm3d_category_name = self.scene.objects[se[i]].category.name()

            if hm3d_category_name in mp3d_category_id:
                # print("sum: ", np.sum(sem_output[sem_output==se[i]])/se[i])
                semantic[semantic==se[i]] = mp3d_category_id[hm3d_category_name]-1
            else :
                semantic[
                    semantic==se[i]
                    ] = 0
    
        # se = list(set(semantic.ravel()))
        # print("semantic: ", se) # []
        # semantic = np.expand_dims(semantic.astype(np.uint8), 2)
        return semantic.astype(np.uint8)

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        self.curr_distance = self._env.get_metrics()['distance_to_goal']

        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def check_llm_shortest_frontier_compatible(self, planner_inputs, frontier_loc_g):

        target_point_map = planner_inputs['map_target']
        pose_pred = planner_inputs['pose_pred']
        full_pose = pose_pred[:3]
        lmb = pose_pred[3:]
        origin = [lmb[2] * self.args.map_resolution / 100.0, lmb[0] * self.args.map_resolution / 100.0, 0.]
        
        # self.map_origin = [-711.378, -481.322] # calculate according to [240, 240] translate to [7.18678, _, 4.88622]
        # self.map_origin = [-2888.622, -3118.678] # calculate according to [240, 240] translate to [7.18678, _, 4.88622]

        # print('target_point_map shape: ', target_point_map.shape) # (480, 480)
        # print('pose_pred: ', pose_pred) # [ 24.  24.   0. 240. 720. 240. 720.]
        # print('full_pose: ', full_pose) # [24. 24.  0.]
        # print('lmb: ', lmb) # [240. 720. 240. 720.]
        # print('origin: ', origin) # [12.0, 12.0, 0.0]

        current_episode = self._env.current_episode

        episode_view_points = [
            view_point.agent_state.position
            for goal in current_episode.goals
            for view_point in goal.view_points
        ]

        # print("===================================================")
        # print('self.args: ', self.args)
        # print('self._env.__dir__(): ', self._env.__dir__())
        # print('self._env.sim.__dir__(): ', self._env.sim.__dir__())
        # print('self._env.sim.agents: ', self._env.sim.agents)
        # print('self._env.current_episode.__dir__(): ', self._env.current_episode.__dir__())
        # # print('len(self._env.current_episode.goals): ', len(self._env.current_episode.goals))
        # print('self._env.current_episode.goals[0].position: ', self._env.current_episode.goals[0].position)
        # print('self._env.current_episode.goals[0].object_name: ', self._env.current_episode.goals[0].object_name)
        # print('self._env.current_episode.goals[1].position: ', self._env.current_episode.goals[1].position)
        # print('self._env.current_episode.goals[1].object_name: ', self._env.current_episode.goals[1].object_name)
        # print('self._env.current_episode.goals[0].__dir__: ', self._env.current_episode.goals[0].__dir__())
        print('self._env.sim.get_agent_state().position: ', self._env.sim.get_agent_state().position)
        # # print('episode_view_points: ', episode_view_points)
        # # print('[goal.position for goal in self._env.current_episode.goals]: ', [goal.position for goal in self._env.current_episode.goals])
        # print('self._env.sim.geodesic_distances: ', self._env.sim.geodesic_distance(self._env.sim.get_agent_state().position, episode_view_points, self._env.current_episode))
        print('self._env.get_metrics(): ', self._env.get_metrics())
        # print("===================================================")

        def sim_map_to_sim_continuous(origin, coords):
            """Converts ground-truth 2D Map coordinates to absolute Habitat
            simulator position.
            """
            # use planner inputs to get origin from lmb to convert local_map coords to global_map coords and then convert to sim coords
            agent_state = self._env.sim.get_agent_state(0)
            agent_position = agent_state.position
            # min_x, min_y = self.map_origin[0] / 100.0, self.map_origin[1] / 100.0
            x, y = coords
            cont_x = x / 20. + origin[0]
            cont_y = (480 - y) / 20. + origin[1]
            # agent_position[0] = cont_y
            # agent_position[2] = cont_x
            agent_position[0] = -0.455 * cont_x - 0.891 * cont_y + 39.487
            agent_position[2] = -0.891 * cont_x + 0.455 * cont_y + 15.362

            return agent_position
        
        def search_valid_points_from_agent_to_frontier(frontier_loc):

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
                plt.savefig("test_viz/frontier_viz_{}.png".format(name))
                plt.close()

            def round_view_of_frontier(frontier_pos, frontier_rot, n=16, name=None):
                for i in range(n):
                    d_rot_rad = i / n * np.pi * 2
                    d_rot_quat = quaternion.from_rotation_vector([0, d_rot_rad, 0])
                    rot_quat = d_rot_quat * frontier_rot
                    # print('frontier_pos: ', frontier_pos)
                    # print('rot_quat: ', rot_quat)
                    frontier_obs = self._env.sim.get_observations_at(frontier_pos, rot_quat)
                    if name is None:
                        display_obs(frontier_obs, "pos_{}_{}".format(frontier_pos, i))
                    else:
                        display_obs(frontier_obs, "{}_{}".format(name, i))

            # from habitat.utils.visualizations import maps
            agent_state = self._env.sim.get_agent_state(0)

            start_point = sim_map_to_sim_continuous(origin, frontier_loc)
            # add observation at this position to verify the validity of the point
            # print('frontier_loc: ', frontier_loc)
            # print('start_point: ', start_point)
            # print('agent_state.rotation: ', agent_state.rotation)

            # round_view_of_frontier(agent_state.position, agent_state.rotation)
            # round_view_of_frontier(start_point, agent_state.rotation, n=16, name="frontier_loc_{}".format(frontier_loc))

            # frontier_obs = self._env.sim.get_observations_at(start_point, agent_state.rotation)
            # display_obs(frontier_obs)
            # print(frontier_obs['rgb'].shape) # (480, 640, 3)
            # print(frontier_obs['depth'].shape) # (480, 640, 1)
            # print(frontier_obs['semantic'].shape) # (480, 640, 1)

            end_point = self._env.sim.get_agent_state(0).position
            sample_points = np.linspace(start_point, end_point, num=100)
            print('start_point: ', start_point)
            # print('end_point: ', end_point)
            # print('sample_points: ', sample_points)
            for sample_point in sample_points:
                is_valid = self._env.sim.pathfinder.is_navigable(sample_point)
                if is_valid:
                    print('valid sample_point: ', sample_point)
                    return sample_point
            print('no valid sample_point')
            return self._env.sim.pathfinder.snap_point(start_point)

        def get_frontier_dis(frontier_loc):
            closest_valid_point = search_valid_points_from_agent_to_frontier(frontier_loc)
            frontier_dis = self._env.sim.geodesic_distance(closest_valid_point, episode_view_points, current_episode)
            return frontier_dis
        
        # frontier_dis_g = self.gt_planner.fmm_dist[frontier_loc_g[0], frontier_loc_g[1]] / 20.0
        print('frontier_loc_g: ', frontier_loc_g)
        frontier_dis_g = get_frontier_dis(frontier_loc_g)
        print('frontier_dis_g: ', frontier_dis_g)

        tpm = len(list(set(target_point_map.ravel()))) -1
        print('tpm: ', tpm)
        for lay in range(tpm):
            # frontier_loc = np.argwhere(target_point_map == lay+1)
            frontier_loc = np.where(target_point_map == lay+1)
            # frontier_distance = self.gt_planner.fmm_dist[frontier_loc[0], frontier_loc[1]] / 20.0
            print("===================================================")
            print('frontier_loc: ', frontier_loc)
            frontier_distance = get_frontier_dis(frontier_loc)
            self.visualize_frontier(planner_inputs, frontier_loc, name="frontier_loc_{}".format(frontier_loc))
            print('frontier_distance: ', frontier_distance)
            print("===================================================")
            if frontier_distance < frontier_dis_g:
                return 0
        return 1
        
    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        dist = self._env.get_metrics()['distance_to_goal']
        if dist < 0.1:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

    def get_done(self, observations):
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
            # print(self._env.get_metrics())
        else:
            done = False
        return done

    def _episode_success(self):
        return self._env.get_metrics()['success']

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def visualize_frontier(self, inputs, frontier_loc, name=None):

        frontier_vis = vu.init_vis_image(self.goal_name, None)

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
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
        sem_map = copy.deepcopy(inputs['sem_map_pred'])

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

        # draw circle around the frontier loc
        frontier_mask = np.zeros_like(edge_mask)
        frontier_mask[frontier_loc[0], frontier_loc[1]] = 1

        frontier_fmb = skimage.draw.circle_perimeter(int(frontier_loc[0]), int(frontier_loc[1]), int(local_w/32))
        frontier_fmb[0][frontier_fmb[0] > local_w-1] = local_w-1
        frontier_fmb[1][frontier_fmb[1] > local_w-1] = local_w-1
        frontier_fmb[0][frontier_fmb[0] < 0] = 0
        frontier_fmb[1][frontier_fmb[1] < 0] = 0
        frontier_mask[frontier_fmb[0], frontier_fmb[1]] = 1
        sem_map[frontier_mask] = 3

        # draw circle around the goal loc
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

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
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        frontier_vis[50:530, 15:655] = self.rgb_vis
        frontier_vis[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1) * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(frontier_vis, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), frontier_vis)
            cv2.waitKey(1)

        if args.print_images:
            if name is None:
                fn = '{}/episodes/thread_{}/eps_{}/frontier-{}-{}-Vis-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                    self.rank, self.episode_no, self.timestep)
            else:
                fn = '{}/episodes/thread_{}/eps_{}/frontier-{}-{}-Vis-{}-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                    self.rank, self.episode_no, self.timestep, name)
            cv2.imwrite(fn, frontier_vis)