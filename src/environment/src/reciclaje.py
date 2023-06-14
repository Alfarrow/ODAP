#!/usr/bin/env python
    #| Establece la pose inicial del robot
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'R2'  # Asegúrate de que 'R2' es el nombre correcto del modelo
            state_msg.pose.position.x = 5.0
            state_msg.pose.position.y = 5.0
            state_msg.pose.position.z = 0.0
            state_msg.pose.orientation.x = 0.0
            state_msg.pose.orientation.y = 0.0
            state_msg.pose.orientation.z = 0.0
            state_msg.pose.orientation.w = 1.0
            state_msg.reference_frame = 'world'
            resp = set_state(state_msg)
            
            if resp.success:  # check if the call was successful
                rospy.loginfo("Model successfully set to initial pose")
            else:
                rospy.logerr("Failed to set the model to initial pose")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        return True