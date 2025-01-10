import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)  # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def calculate_angle(point1, point2, point3):
    vector1 = np.subtract(point1, point2)
    vector2 = np.subtract(point3, point2)
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_theta = dot_product / norm_product
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def calculate_finger_angles(hand_landmarks):
    # Calculate angles between the palm (wrist) and fingers
    angles = {}
    for finger, (base, mid, tip) in {
        "Index": (0, 5, 8),
        "Middle": (0, 9, 12),
        "Ring": (0, 13, 16),
        "Pinky": (0, 17, 20)
    }.items():
        if hand_landmarks.landmark[base] and hand_landmarks.landmark[mid] and hand_landmarks.landmark[tip]:
            base_point = np.array([hand_landmarks.landmark[base].x, hand_landmarks.landmark[base].y])
            mid_point = np.array([hand_landmarks.landmark[mid].x, hand_landmarks.landmark[mid].y])
            tip_point = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y])
            angles[finger] = calculate_angle(base_point, mid_point, tip_point)
    return angles


def draw_styled_landmarks(image, results):

    def get_point(landmark_index):
        if results.pose_landmarks:
            if results.pose_landmarks.landmark[landmark_index].visibility > 0.5:
                return np.array([
                    results.pose_landmarks.landmark[landmark_index].x,
                    results.pose_landmarks.landmark[landmark_index].y
                ])
            else:
                return None
        return None

    # Numbering
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            cv2.putText(image, str(idx), (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if results.right_hand_landmarks:
        for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
            cv2.putText(image, str(idx), (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if results.left_hand_landmarks:
        for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
            cv2.putText(image, str(idx), (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Get Points
    right_shoulder = get_point(11)
    right_elbow = get_point(13)
    right_wrist = get_point(15)
    left_shoulder = get_point(12)
    left_elbow = get_point(14)
    left_wrist = get_point(16)
    right_hip = get_point(23)
    left_hip = get_point(24)
    right_hand = get_point(21)
    left_hand = get_point(22)

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # LEFT Hand Finger Angles / *** Text Mirrored ***
    if results.right_hand_landmarks:
        right_finger_angles = calculate_finger_angles(results.right_hand_landmarks)
        for finger, angle in right_finger_angles.items():
            cv2.putText(image, f"LEFT {finger}: {angle:.2f}",
                        (10, 200 + 20 * list(right_finger_angles.keys()).index(finger)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # RIGHT Hand Finger Angles / *** Text Mirrored ***
    if results.left_hand_landmarks:
        left_finger_angles = calculate_finger_angles(results.left_hand_landmarks)
        for finger, angle in left_finger_angles.items():
            cv2.putText(image, f"RIGHT {finger}: {angle:.2f}",
                        (10, 300 + 20 * list(left_finger_angles.keys()).index(finger)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Error
    if right_shoulder is None or right_elbow is None or right_wrist is None:
        right_arm_angle = "RIGHT ARM: point missing!"
    else:
        right_arm_angle = f"RIGHT ARM: {calculate_angle(right_shoulder, right_elbow, right_wrist):.2f}"

    if left_shoulder is None or left_elbow is None or left_wrist is None:
        left_arm_angle = "LEFT ARM: point missing!"
    else:
        left_arm_angle = f"LEFT ARM: {calculate_angle(left_shoulder, left_elbow, left_wrist):.2f}"

    if right_hip is None or right_shoulder is None or right_elbow is None:
        right_torso_angle = "RIGHT TORSO: point missing!"
    else:
        right_torso_angle = f"RIGHT TORSO: {calculate_angle(right_hip, right_shoulder, right_elbow):.2f}"

    if left_hip is None or left_shoulder is None or left_elbow is None:
        left_torso_angle = "LEFT TORSO: point missing!"
    else:
        left_torso_angle = f"LEFT TORSO: {calculate_angle(left_hip, left_shoulder, left_elbow):.2f}"

    if right_elbow is None or right_wrist is None or right_hand is None:
        right_wrist_angle = "RIGHT WRIST: point missing!"
    else:
        right_wrist_angle = f"RIGHT WRIST: {calculate_angle(right_elbow, right_wrist, right_hand):.2f}"

    if left_elbow is None or left_wrist is None or left_hand is None:
        left_wrist_angle = "LEFT WRIST: point missing!"
    else:
        left_wrist_angle = f"LEFT WRIST: {calculate_angle(left_elbow, left_wrist, left_hand):.2f}"

    # Display
    cv2.putText(image, str(left_wrist_angle), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, str(right_wrist_angle), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, str(right_arm_angle), (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, str(left_arm_angle), (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, str(right_torso_angle), (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, str(left_torso_angle), (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # *** Signals ***
    # Touch
    if results.right_hand_landmarks and results.left_hand_landmarks:
        right_hand_points = [
            results.right_hand_landmarks.landmark[i] for i in [8, 12, 16]
        ]
        left_hand_points = results.left_hand_landmarks.landmark

        for right_point in right_hand_points:
            for left_point in left_hand_points:
                if (abs(right_point.x - left_point.x) < 0.07 and
                        abs(right_point.y - left_point.y) < 0.07 and
                        abs(right_point.z - left_point.z) < 0.07):  # Adjust threshold
                    cv2.putText(image, "touch!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Crossed Close to Body
    if right_shoulder is not None and left_shoulder is not None:
        chest_top = (right_shoulder[1] + left_shoulder[1]) / 2
        chest_bottom = chest_top + 0.4

        if right_hand is not None:
            right_hand_close = ((abs(right_hand[0] - right_shoulder[0]) < 0.08 or
                                abs(right_hand[0] - left_shoulder[0]) < 0.08) and
                                chest_top < right_hand[1] < chest_bottom)
            if right_hand_close:
                cv2.putText(image, "RIGHT close!", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if left_hand is not None:
            left_hand_close = ((abs(left_hand[0] - right_shoulder[0]) < 0.08 or
                                abs(left_hand[0] - left_shoulder[0]) < 0.08) and
                               chest_top < left_hand[1] < chest_bottom)
            if left_hand_close:
                cv2.putText(image, "LEFT close!", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Right or Left
    # Midpoints
    if right_shoulder is not None and left_shoulder is not None and right_hip is not None and left_hip is not None:
        upper_midpoint = (right_shoulder + left_shoulder) / 2
        lower_midpoint = (right_hip + left_hip) / 2
        # Visuals for R/L
        cv2.line(image,
                 (int(upper_midpoint[0] * image.shape[1]), int(upper_midpoint[1] * image.shape[0])),
                 (int(lower_midpoint[0] * image.shape[1]), int(lower_midpoint[1] * image.shape[0])),
                 (0, 255, 0), 2)
        # Draw star markers at the midpoints
        cv2.drawMarker(image,
                       (int(upper_midpoint[0] * image.shape[1]), int(upper_midpoint[1] * image.shape[0])),
                       (0, 0, 255), cv2.MARKER_STAR, markerSize=20)
        cv2.drawMarker(image,
                       (int(lower_midpoint[0] * image.shape[1]), int(lower_midpoint[1] * image.shape[0])),
                       (0, 0, 255), cv2.MARKER_STAR, markerSize=20)

        def determine_side(point, line_start, line_end):
            line_vector = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
            point_vector = np.array([point[0] - line_start[0], point[1] - line_start[1]])
            cross_product = np.cross(line_vector, point_vector)
            if cross_product > 0:
                return "RIGHT"
            elif cross_product < 0:
                return "LEFT"
            else:
                return "ON THE LINE"

        if right_hand is not None:
            right_hand_side = determine_side(right_hand, upper_midpoint, lower_midpoint)
            if right_hand_side == "RIGHT":
                cv2.putText(image, "RIGHT on L!", (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif right_hand_side == "RIGHT":
                cv2.putText(image, "RIGHT on R!", (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if left_hand is not None:
            left_hand_side = determine_side(left_hand, upper_midpoint, lower_midpoint)
            if left_hand_side == "LEFT":
                cv2.putText(image, "LEFT on R!", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif left_hand_side == "LEFT":
                cv2.putText(image, "LEFT on L!", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Nothing / In / Point
        right_sign = "NOTHING"
        left_sign = "NOTHING"
        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            right_arm_angle_value = calculate_angle(right_shoulder, right_elbow, right_wrist)
            if (130 <= right_arm_angle_value <= 180 and right_torso_angle != "RIGHT TORSO: point missing!" and
                    60 <= float(right_torso_angle.split(': ')[1]) <= 120):
                right_sign = "POINT"
            elif (130 <= right_arm_angle_value <= 180 and right_torso_angle != "RIGHT TORSO: point missing!" and
                  15 <= float(right_torso_angle.split(': ')[1]) < 80):
                right_sign = "IN"

        if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
            left_arm_angle_value = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if (130 <= left_arm_angle_value <= 180 and left_torso_angle != "LEFT TORSO: point missing!" and
                    60 <= float(left_torso_angle.split(': ')[1]) <= 120):
                left_sign = "POINT"
            elif (130 <= left_arm_angle_value <= 180 and left_torso_angle != "LEFT TORSO: point missing!" and
                  15 <= float(left_torso_angle.split(': ')[1]) < 80):
                left_sign = "IN"

            # Display Nothing / In / Point
            cv2.putText(image, f"RIGHT: {right_sign}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"LEFT: {left_sign}", (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # OUT if not END GAME
        right_out = "NOT OUT"
        left_out = "NOT OUT"

        if (right_shoulder is not None and left_shoulder is not None and
                right_elbow is not None and left_elbow is not None and
                right_wrist is not None and left_wrist is not None and
                right_hand is not None and left_hand is not None):

            right_arm_angle_value = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_arm_angle_value = calculate_angle(left_shoulder, left_elbow, left_wrist)

            wrist_distance = np.linalg.norm(np.array(right_wrist) - np.array(left_wrist))
            wrists_crossed = wrist_distance < 0.15

            chest_top = (right_shoulder[1] + left_shoulder[1]) / 2
            chest_bottom = chest_top + 0.4
            right_hand_close = (abs(right_hand[0] - left_shoulder[0]) < 0.1 and
                                chest_top < right_hand[1] < chest_bottom)
            left_hand_close = (abs(left_hand[0] - right_shoulder[0]) < 0.1 and
                               chest_top < left_hand[1] < chest_bottom)

            arms_bent = (0 <= right_arm_angle_value <= 50 and
                         0 <= left_arm_angle_value <= 50)

            if left_hand_close and right_hand_close and wrists_crossed and arms_bent:
                cv2.putText(image, "END GAME!", (10, 620),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # Visualisation
                cv2.line(image,
                         (int(right_wrist[0] * image.shape[1]), int(right_wrist[1] * image.shape[0])),
                         (int(left_wrist[0] * image.shape[1]), int(left_wrist[1] * image.shape[0])),
                         (255, 0, 255), 2)

            else:
                if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
                    right_arm_angle_value = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    if right_torso_angle != "RIGHT TORSO: point missing!":
                        right_torso_value = float(right_torso_angle.split(': ')[1])
                        # Check if arm is straight up (close to 180 degrees) and torso angle is upright
                        if (0 <= right_arm_angle_value <= 40) and (0 <= right_torso_value <= 60):
                            # Check if fingers are extended (using existing finger angles)
                            if results.right_hand_landmarks:
                                right_finger_angles = calculate_finger_angles(results.right_hand_landmarks)
                                if all(155 <= angle <= 180 for angle in right_finger_angles.values()):
                                    right_out = "OUT"

                if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
                    left_arm_angle_value = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    if left_torso_angle != "LEFT TORSO: point missing!":
                        left_torso_value = float(left_torso_angle.split(': ')[1])
                        # Check if arm is straight up (close to 180 degrees) and torso angle is upright
                        if (0 <= left_arm_angle_value <= 40) and (0 <= left_torso_value <= 60):
                            # Check if fingers are extended (using existing finger angles)
                            if results.left_hand_landmarks:
                                left_finger_angles = calculate_finger_angles(results.left_hand_landmarks)
                                if all(60 <= angle <= 180 for angle in left_finger_angles.values()):
                                    left_out = "OUT"

                if right_out == "OUT" and left_out == "OUT":
                    cv2.putText(image, "OUT!", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def runTesting():
    cap = cv2.VideoCapture(1)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Mirror the frame
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    runTesting()
