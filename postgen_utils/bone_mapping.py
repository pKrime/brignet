
class HumanLimb:
    def __str__(self):
        return self.__class__.__name__ + ' ' + ', '.join(["{0}: {1}".format(k, v) for k, v in self.__dict__.items()])

    def __getitem__(self, item):
        return getattr(self, item, None)

    def items(self):
        return self.__dict__.items()


class HumanSpine(HumanLimb):
    def __init__(self, head='', neck='', spine2='', spine1='', spine='', hips=''):
        self.head = head
        self.neck = neck
        self.spine2 = spine2
        self.spine1 = spine1
        self.spine = spine
        self.hips = hips


class HumanArm(HumanLimb):
    def __init__(self, shoulder='', arm='', forearm='', hand=''):
        self.shoulder = shoulder
        self.arm = arm
        self.arm_twist = None
        self.forearm = forearm
        self.forearm_twist = None
        self.hand = hand


class HumanLeg(HumanLimb):
    def __init__(self, upleg='', leg='', foot='', toe=''):
        self.upleg = upleg
        self.upleg_twist = None
        self.leg = leg
        self.leg_twist = None
        self.foot = foot
        self.toe = toe


class HumanFingers(HumanLimb):
    def __init__(self, thumb=[''] * 3, index=[''] * 3, middle=[''] * 3, ring=[''] * 3, pinky=[''] * 3):
        self.thumb = thumb
        self.index = index
        self.middle = middle
        self.ring = ring
        self.pinky = pinky


class HumanSkeleton:
    spine = None

    left_arm = None
    right_arm = None
    left_leg = None
    right_leg = None

    left_fingers = None
    right_fingers = None

    def conversion_map(self, target_skeleton):
        """Return a dictionary that maps skeleton bone names to target bone names
        >>> rigify = RigifySkeleton()
        >>> rigify.conversion_map(MixamoSkeleton())
        {'DEF-spine.006': 'Head', 'DEF-spine.004': 'Neck', 'DEF-spine.003'...
        """
        bone_map = dict()

        def bone_mapping(attr, limb, bone_name):
            target_limbs = getattr(target_skeleton, attr, None)
            if not target_limbs:
                return

            trg_name = target_limbs[limb]

            if trg_name:
                bone_map[bone_name] = trg_name

        for limb_name, bone_name in self.spine.items():
            bone_mapping('spine', limb_name, bone_name)

        for limb_name, bone_name in self.left_arm.items():
            bone_mapping('left_arm', limb_name, bone_name)

        for limb_name, bone_name in self.right_arm.items():
            bone_mapping('right_arm', limb_name, bone_name)

        for limb_name, bone_name in self.left_leg.items():
            bone_mapping('left_leg', limb_name, bone_name)

        for limb_name, bone_name in self.right_leg.items():
            bone_mapping('right_leg', limb_name, bone_name)

        def fingers_mapping(src_fingers, trg_fingers):
            for finger, bone_names in src_fingers.items():
                trg_bone_names = trg_fingers[finger]

                assert len(bone_names) == len(trg_bone_names)
                for bone, trg_bone in zip(bone_names, trg_bone_names):
                    bone_map[bone] = trg_bone

        trg_fingers = target_skeleton.left_fingers
        fingers_mapping(self.left_fingers, trg_fingers)

        trg_fingers = target_skeleton.right_fingers
        fingers_mapping(self.right_fingers, trg_fingers)

        return bone_map


class RigifySkeleton(HumanSkeleton):
    def __init__(self):
        self.spine = HumanSpine(
            head='DEF-spine.006',
            neck='DEF-spine.004',
            spine2='DEF-spine.003',
            spine1='DEF-spine.002',
            spine='DEF-spine.001',
            hips='DEF-spine'
        )

        for side, side_letter in zip(('left', 'right'), ('L', 'R')):
            arm = HumanArm(shoulder="DEF-shoulder.{0}".format(side_letter),
                           arm="DEF-upper_arm.{0}".format(side_letter),
                           forearm="DEF-forearm.{0}".format(side_letter),
                           hand="DEF-hand.{0}".format(side_letter))

            arm.arm_twist = arm.arm + ".001"
            arm.forearm_twist = arm.forearm + ".001"
            setattr(self, side + "_arm", arm)

            fingers = HumanFingers(
                thumb=["DEF-thumb.{1:02d}.{0}".format(side_letter, i) for i in range(1, 4)],
                index=["DEF-f_index.{1:02d}.{0}".format(side_letter, i) for i in range(1, 4)],
                middle=["DEF-f_middle.{1:02d}.{0}".format(side_letter, i) for i in range(1, 4)],
                ring=["DEF-f_ring.{1:02d}.{0}".format(side_letter, i) for i in range(1, 4)],
                pinky=["DEF-f_pinky.{1:02d}.{0}".format(side_letter, i) for i in range(1, 4)],
            )
            setattr(self, side + "_fingers", fingers)

            leg = HumanLeg(upleg="DEF-thigh.{0}".format(side_letter),
                           leg="DEF-shin.{0}".format(side_letter),
                           foot="DEF-foot.{0}".format(side_letter),
                           toe="DEF-toe.{0}".format(side_letter))

            leg.upleg_twist = leg.upleg + ".001"
            leg.leg_twist = leg.leg + ".001"
            setattr(self, side + "_leg", leg)


class RigifyMeta(HumanSkeleton):
    def __init__(self):
        self.spine = HumanSpine(
            head='spine.006',
            neck='spine.004',
            spine2='spine.003',
            spine1='spine.002',
            spine='spine.001',
            hips='spine'
        )

        side = 'L'
        self.left_arm = HumanArm(shoulder="shoulder.{0}".format(side),
                                 arm="upper_arm.{0}".format(side),
                                 forearm="forearm.{0}".format(side),
                                 hand="hand.{0}".format(side))

        self.left_fingers = HumanFingers(
            thumb=["thumb.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            index=["f_index.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            middle=["f_middle.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            ring=["f_ring.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            pinky=["f_pinky.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
        )

        self.left_leg = HumanLeg(upleg="thigh.{0}".format(side),
                                 leg="shin.{0}".format(side),
                                 foot="foot.{0}".format(side),
                                 toe="toe.{0}".format(side))

        side = 'R'
        self.right_arm = HumanArm(shoulder="shoulder.{0}".format(side),
                                  arm="upper_arm.{0}".format(side),
                                  forearm="forearm.{0}".format(side),
                                  hand="hand.{0}".format(side))

        self.right_fingers = HumanFingers(
            thumb=["thumb.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            index=["f_index.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            middle=["f_middle.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            ring=["f_ring.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
            pinky=["f_pinky.{1:02d}.{0}".format(side, i) for i in range(1, 4)],
        )

        self.right_leg = HumanLeg(upleg="thigh.{0}".format(side),
                                  leg="shin.{0}".format(side),
                                  foot="foot.{0}".format(side),
                                  toe="toe.{0}".format(side))


class UnrealSkeleton(HumanSkeleton):
    def __init__(self):
        self.spine = HumanSpine(
            head='head',
            neck='neck_01',
            spine2='spine_03',
            spine1='spine_02',
            spine='spine_01',
            hips='pelvis'
        )

        for side, side_letter in zip(('left', 'right'), ('_l', '_r')):
            arm = HumanArm(shoulder="clavicle" + side_letter,
                           arm="upperarm" + side_letter,
                           forearm="lowerarm" + side_letter,
                           hand="hand" + side_letter)

            arm.arm_twist = "upperarm_twist_01" + side_letter
            arm.forearm_twist = "lowerarm_twist_01" + side_letter
            setattr(self, side + "_arm", arm)

            fingers = HumanFingers(
                    thumb=["thumb_{0:02d}{1}".format(i, side_letter) for i in range(1, 4)],
                    index=["index_{0:02d}{1}".format(i, side_letter) for i in range(1, 4)],
                    middle=["middle_{0:02d}{1}".format(i, side_letter) for i in range(1, 4)],
                    ring=["ring_{0:02d}{1}".format(i, side_letter) for i in range(1, 4)],
                    pinky=["pinky_{0:02d}{1}".format(i, side_letter) for i in range(1, 4)],
                )
            setattr(self, side + "_fingers", fingers)

            leg = HumanLeg(upleg="thigh{0}".format(side_letter),
                           leg="calf{0}".format(side_letter),
                           foot="foot{0}".format(side_letter),
                           toe="ball{0}".format(side_letter))

            leg.upleg_twist = "thigh_twist_01" + side_letter
            leg.leg_twist = "calf_twist_01" + side_letter
            setattr(self, side + "_leg", leg)
