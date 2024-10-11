import csv
from math import radians, cos, sin, sqrt, atan2  # use for Heuristic Approaches
from collections import deque  # use for undirected (blind) brute-force approaches
import heapq
import time  # for measuring the time of the search
import sys  # for memory usage calculation (optional)
import matplotlib.pyplot as plt  # for plotting the route

# Function to build the adjacency list from the text file
def build_graph(adjacency_file):
    graph = {}
    with open(adjacency_file, 'r') as file:
        for line in file:
            city1, city2 = line.strip().split()
            if city1 not in graph:
                graph[city1] = []
            if city2 not in graph:
                graph[city2] = []
            if city2 not in graph[city1]:
                graph[city1].append(city2)
            if city1 not in graph[city2]:
                graph[city2].append(city1)
    return graph

# Function to load coordinates from the CSV file
def load_coordinates(coordinate_file):
    coordinates = {}
    with open(coordinate_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            city = row[0]  # First column is the city name
            lat = float(row[1])  # Second column is latitude
            lon = float(row[2])  # Third column is longitude
            coordinates[city] = (lat, lon)
    return coordinates

# Compute the straight-line distance between two cities, used for best-first search and A*
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Example to calculate distance between two cities
def get_city_distance(city1, city2, coordinates):
    lat1, lon1 = coordinates[city1]
    lat2, lon2 = coordinates[city2]
    return calculate_distance(lat1, lon1, lat2, lon2)

# Function to calculate the total distance of a path
def calculate_total_distance(path, coordinates):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += get_city_distance(path[i], path[i+1], coordinates)
    return total_distance

# Function to plot the route on a 2D map
def plot_route(path, coordinates):
    lats = [coordinates[city][0] for city in path]
    lons = [coordinates[city][1] for city in path]

    plt.figure(figsize=(10, 8))
    plt.plot(lons, lats, marker="o", color="b", linestyle="-")
    
    for i, city in enumerate(path):
        plt.text(lons[i], lats[i], city, fontsize=12, ha='right')
    
    plt.title(f"Route from {path[0]} to {path[-1]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# Optional: Function to calculate memory usage (if needed)
def calculate_memory_usage(*args):
    total_memory = sum(sys.getsizeof(arg) for arg in args)
    return total_memory
def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    iteration_count = 0
    
    while queue:
        iteration_count += 1
        current_node, path = queue.popleft()
        # print(f"Processing node: {current_node}, Path: {path}, Queue length: {len(queue)}")
        
        if current_node == goal:
            # print(f"Goal found in {iteration_count} iterations using BFS")
            return path, iteration_count
        
        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
                visited.add(neighbor)  # Mark as visited when enqueued to avoid duplicates

    return None, iteration_count  # Return iteration count even if the goal is not found


# Depth-First Search
def dfs(graph, start, goal, visited=None, path=None, iterations=0):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]
    
    iterations += 1  # Increment on each node visit
    if start == goal:
        return path, iterations
    
    visited.add(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result, iterations = dfs(graph, neighbor, goal, visited, path + [neighbor], iterations)
            if result:
                return result, iterations
    return None, iterations


# ID-DFS search
def iddfs(graph, start, goal, max_depth):
    total_iterations = 0
    for depth in range(max_depth + 1):
        result, iterations = dls(graph, start, goal, depth)
        total_iterations += iterations
        if result:
            return result, total_iterations
    return None, total_iterations

def dls(graph, node, goal, depth, path=None, iterations=0):
    if path is None:
        path = [node]
    
    iterations += 1  # Increment on each node visit
    if depth == 0 and node == goal:
        return path, iterations
    if depth > 0:
        for neighbor in graph[node]:
            result, iterations = dls(graph, neighbor, goal, depth - 1, path + [neighbor], iterations)
            if result:
                return result, iterations
    return None, iterations


# Heuristic Approaches

# Best-First Search
def best_first_search(graph, start, goal, coordinates):
    def heuristic(city1, city2):
        return get_city_distance(city1, city2, coordinates)

    queue = []
    heapq.heappush(queue, (0, start, [start]))  # (heuristic cost, current city, path)
    visited = set()
    iterations = 0  # Initialize iteration counter

    while queue:
        _, current_node, path = heapq.heappop(queue)
        iterations += 1  # Increment on each node visit

        if current_node in visited:
            continue

        if current_node == goal:
            return path, iterations

        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                priority = heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor, path + [neighbor]))

    return None, iterations

# A* Search implementation
def a_star(graph, start, goal, coordinates):
    def heuristic(city1, city2):
        return get_city_distance(city1, city2, coordinates)

    queue = []
    heapq.heappush(queue, (0, start, [start]))  # (total cost, current city, path)
    visited = set()
    iterations = 0  # Initialize iteration counter

    while queue:
        cost, current_node, path = heapq.heappop(queue)
        iterations += 1  # Increment on each node visit

        if current_node in visited:
            continue

        if current_node == goal:
            return path, iterations

        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                g = cost + get_city_distance(current_node, neighbor, coordinates)
                h = heuristic(neighbor, goal)
                heapq.heappush(queue, (g + h, neighbor, path + [neighbor]))

    return None, iterations


def main():
    graph = build_graph('Adjacencies.txt')
    coordinates = load_coordinates('coordinates.csv')
    
    while True:
        start = input("Enter starting city: ")
        goal = input("Enter destination city: ")
        
        if start not in graph or goal not in graph:
            print("Invalid cities. Please enter valid city names.")
            continue
        
        print("Choose a search method:")
        print("1: BFS")
        print("2: DFS")
        print("3: ID-DFS")
        print("4: Best-First Search")
        print("5: A* Search")
        
        method = input("Enter the number of the search method: ")
        
        start_time = time.perf_counter()  # Use time.perf_counter() for precise time measurement
        
        if method == "1":
            path, iterations = bfs(graph, start, goal)
        elif method == "2":
            path, iterations = dfs(graph, start, goal)
        elif method == "3":
            path, iterations = iddfs(graph, start, goal, max_depth=5)  # Adjust depth as needed
        elif method == "4":
            path, iterations = best_first_search(graph, start, goal, coordinates)
        elif method == "5":
            path, iterations = a_star(graph, start, goal, coordinates)
        else:
            print("Invalid method.")
            continue
        
        end_time = time.perf_counter()  # End time measurement
        total_time = end_time - start_time
        
        if path:
            print("Path found:", " -> ".join(path))
            print(f"Time taken to find the route: {total_time:.6f} seconds")
            print(f"Total iterations: {iterations}")
            
            total_distance = calculate_total_distance(path, coordinates)
            print(f"Total distance for the route: {total_distance:.2f} km")
            
            plot_route(path, coordinates)  # Plot the route

            # Optional: Calculate memory usage
            memory_used = calculate_memory_usage(graph, coordinates, path)
            print(f"Memory used: {memory_used / 1024:.2f} KB")
            
        else:
            print("No path found.")
        
        # Ask the user if they want to perform another calculation
        repeat = input("Would you like to calculate another distance? (yes/no): ").strip().lower()
        if repeat != 'yes':
            break 


if __name__ == "__main__":
    main()
