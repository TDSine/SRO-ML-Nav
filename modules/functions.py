# Import modules
import socket
import random
import pickle

# Function 1
def send_command(command):
    host = 'localhost'  # The server's hostname or IP address
    port = 8052         # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(command.encode('utf-8'))

        # Now receive data back from the server
        response = s.recv(1024)  # Adjust buffer size as needed 
        response_str = response.decode('utf-8')

        # Extracting the coordinates from the response
        # Assuming the response format is "X: [value], Y: [value]"
        coords = response_str.split(',')
        x_coord = int(round(float(coords[0])))
        y_coord = int(round(float(coords[1])))

        return x_coord, y_coord
    
def update_probabilities(action_probabilities, last_action, reward, learning_rate=0.05):
    for action in action_probabilities:
        if action == last_action:
            action_probabilities[action] += learning_rate * reward
        else:
            action_probabilities[action] -= (learning_rate * reward) / (len(action_probabilities) - 1)

    # Normalize the probabilities
    total = sum(action_probabilities.values())
    for action in action_probabilities:
        action_probabilities[action] /= total

    return action_probabilities

def select_action(action_probabilities):
    actions, probabilities = zip(*action_probabilities.items())
    action = random.choices(actions, weights=probabilities, k=1)[0]
    return action

def calculate_reward(current_position, target_position, new_position, hit_obstacle, action_history):

    # Define a large penalty for hitting an obstacle
    obstacle_penalty = -75
    repetitive_action_penalty = -0.5

    # If an obstacle is hit, return the large penalty
    if hit_obstacle:
        return obstacle_penalty

    target_x, target_y = target_position
    current_x, current_y = current_position
    new_x, new_y = new_position

    # Check for repetitive actions
    try:
        if len(action_history) > 2 and action_history[-1] == action_history[-3]:
            return repetitive_action_penalty
    except Exception:
        pass

    # Calculate the Euclidean distance to the target from the current and new positions
    distance_to_target = ((target_x - current_x)**2 + (target_y - current_y)**2)**0.5
    new_distance_to_target = ((target_x - new_x)**2 + (target_y - new_y)**2)**0.5

    # Set the reward structure
    if (new_x, new_y) == target_position:
        return 100  # Large reward for reaching the target
    elif new_distance_to_target < distance_to_target:
        return 1  # Positive reward for moving closer
    elif new_distance_to_target > distance_to_target:
        return -3  # Negative reward for moving away
    else:
        return -0.05  # Small penalty for any move

def reset_action_probability(action_probabilities, action_to_reset, reset_value=0.1):
    num_actions = len(action_probabilities)
    delta = (action_probabilities[action_to_reset] - reset_value) / (num_actions - 1)

    for action in action_probabilities:
        if action == action_to_reset:
            action_probabilities[action] = reset_value
        else:
            action_probabilities[action] += delta

    return action_probabilities

def record_action_outcome(action_history, start_position, target_position, action, reward):
    key = (start_position, target_position)
    if key not in action_history:
        action_history[key] = []

    action_history[key].append((action, reward))

def select_action_based_on_history(action_probabilities, action_history, current_position, target_position, learning_rate=0.01):
    key = (current_position, target_position)

    if key in action_history:
        # Create a copy of the action_probabilities to modify
        updated_probabilities = action_probabilities.copy()

        for action, reward in action_history[key]:
            if action in updated_probabilities:
                # Update the probability based on reward
                # Increase for positive rewards, decrease for negative
                updated_probabilities[action] += learning_rate * reward

                # Ensure probability is within valid range [0, 1]
                updated_probabilities[action] = max(0, min(updated_probabilities[action], 1))

        # Normalize the probabilities
        total = sum(updated_probabilities.values())
        for action in updated_probabilities:
            updated_probabilities[action] /= total

        # Select an action based on the updated probabilities
        actions, probabilities = zip(*updated_probabilities.items())
        selected_action = random.choices(actions, weights=probabilities, k=1)[0]
    else:
        # If no history, fall back to the original method
        selected_action = select_action(action_probabilities)

    return selected_action

def export_action_history_to_text(action_history, filename):
    with open(filename, 'w') as file:
        for key, actions in action_history.items():
            start_position, target_position = key
            file.write(f"Start Position: {start_position}, Target Position: {target_position}\n")
            for action, reward in actions:
                file.write(f"\tAction: {action}, Reward: {reward}\n")
            file.write("\n")

#Function 3
def save_action_history(action_history, filename):
    with open(filename, 'wb') as file:  # 'wb' mode for writing in binary
        pickle.dump(action_history, file)

def load_action_history(filename):
    try:
        with open(filename, 'rb') as file:  # 'rb' mode for reading in binary
            return pickle.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if file doesn't exist
    
def save_successful_routes(successful_routes, filename):
    with open(filename, 'wb') as file:  # 'wb' mode for writing in binary
        pickle.dump(successful_routes, file)

def load_successful_routes(filename):
    try:
        with open(filename, 'rb') as file:  # 'rb' mode for reading in binary
            return pickle.load(file)
    except FileNotFoundError:
        return []  # Return an empty dictionary if file doesn't exist
    
def save_obstacle_points(obstacle_points, filename):
    with open(filename, 'wb') as file:  # 'wb' mode for writing in binary
        pickle.dump(obstacle_points, file)

def load_obstacle_points(filename):
    try:
        with open(filename, 'rb') as file:  # 'rb' mode for reading in binary
            return pickle.load(file)
    except FileNotFoundError:
        return set()  # Return an empty set if file doesn't exist
    
def avoid_obstacles(action_probabilities, current_position, detected_obstacles):
    for action in action_probabilities.keys():
        new_position = predict_new_position(current_position, action)
        if new_position in detected_obstacles:
            action_probabilities[action] *= 0.1  # Significantly reduce the probability

    # Normalize probabilities
    total_prob = sum(action_probabilities.values())
    if total_prob > 0:
        action_probabilities = {action: prob / total_prob for action, prob in action_probabilities.items()}
    return action_probabilities

def predict_new_position(position, action):
    x, y = position
    if action == "UP":
        return x, y - 1
    elif action == "DOWN":
        return x, y + 1
    elif action == "LEFT":
        return x - 1, y
    elif action == "RIGHT":
        return x + 1, y
    return position  # Return the original position if action is unrecognized

def analyze_successful_routes(successful_routes):
    action_patterns = {}
    for route in successful_routes:
        for index, action in enumerate(route):
            if index not in action_patterns:
                action_patterns[index] = {}
            if action not in action_patterns[index]:
                action_patterns[index][action] = 0
            action_patterns[index][action] += 1
    return action_patterns

def adjust_probabilities_based_on_success(action_probabilities, action_patterns, current_step):
    if current_step in action_patterns:
        for action, count in action_patterns[current_step].items():
            if action in action_probabilities:
                action_probabilities[action] += (0.05 * count)

    # Ensure probabilities sum to 1
    total_prob = sum(action_probabilities.values())
    action_probabilities = {a: p / total_prob for a, p in action_probabilities.items()}
    return action_probabilities
