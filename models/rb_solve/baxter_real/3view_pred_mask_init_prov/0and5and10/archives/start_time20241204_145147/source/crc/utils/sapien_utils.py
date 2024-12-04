def find_joint_by_name(robot, name):
    for j in robot.get_active_joints():
        if j.name == name:
            return j
    raise RuntimeError()


def find_link_by_name(robot, name):
    for l in robot.get_links():
        if l.name == name:
            return l
    raise RuntimeError()
