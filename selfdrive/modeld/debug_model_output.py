#!/usr/bin/env python3
"""
Debug script to understand the structure of model output messages
"""

import cereal.messaging as messaging
from cereal import log
import time

def debug_model_output():
    """Debug the structure of modelV2 messages"""
    print("Model Output Structure Debug")
    print("=" * 40)

    # Set up messaging
    sm = messaging.SubMaster(['modelV2'])

    print("Waiting for modelV2 messages...")
    print("Press Ctrl+C to stop.")
    print()

    try:
        while True:
            # Update messaging
            sm.update(timeout=1000)  # 1 second timeout

            if sm.updated['modelV2']:
                model_output = sm['modelV2']

                print("=== ModelV2 Message Structure ===")
                print(f"Message type: {type(model_output)}")
                print(f"Available fields: {dir(model_output)}")
                print()

                # Check for lane lines
                if hasattr(model_output, 'laneLines'):
                    lane_lines = model_output.laneLines
                    print(f"Lane Lines: {type(lane_lines)}")

                    # Try to access lane lines as a list
                    try:
                        lane_lines_list = list(lane_lines)
                        print(f"Lane Lines length: {len(lane_lines_list)}")

                        if len(lane_lines_list) > 0:
                            first_lane = lane_lines_list[0]
                            print(f"First lane line type: {type(first_lane)}")
                            print(f"First lane line fields: {dir(first_lane)}")

                            # Try to access x, y coordinates directly
                            if hasattr(first_lane, 'x'):
                                x_val = first_lane.x
                                print(f"First lane line x: {x_val} (type: {type(x_val)})")
                                try:
                                    x_float = float(x_val)
                                    print(f"Converted x to float: {x_float}")
                                except Exception as e:
                                    print(f"Error converting x to float: {e}")

                            if hasattr(first_lane, 'y'):
                                y_val = first_lane.y
                                print(f"First lane line y: {y_val} (type: {type(y_val)})")
                                try:
                                    y_float = float(y_val)
                                    print(f"Converted y to float: {y_float}")
                                except Exception as e:
                                    print(f"Error converting y to float: {e}")

                            if hasattr(first_lane, 'z'):
                                z_val = first_lane.z
                                print(f"First lane line z: {z_val} (type: {type(z_val)})")

                    except Exception as e:
                        print(f"Error converting lane lines to list: {e}")
                print()

                # Check for lane probabilities
                if hasattr(model_output, 'laneLineProbs'):
                    lane_probs = model_output.laneLineProbs
                    print(f"Lane Probabilities: {type(lane_probs)}")
                    try:
                        lane_probs_list = list(lane_probs)
                        print(f"Lane Probabilities length: {len(lane_probs_list)}")
                        if len(lane_probs_list) > 0:
                            print(f"First probability: {lane_probs_list[0]}")
                    except Exception as e:
                        print(f"Error accessing lane probabilities: {e}")
                print()

                # Check for road edges
                if hasattr(model_output, 'roadEdges'):
                    road_edges = model_output.roadEdges
                    print(f"Road Edges: {type(road_edges)}")
                    try:
                        road_edges_list = list(road_edges)
                        print(f"Road Edges length: {len(road_edges_list)}")
                        if len(road_edges_list) > 0:
                            first_edge = road_edges_list[0]
                            print(f"First road edge type: {type(first_edge)}")
                            try:
                                edge_points = list(first_edge)
                                print(f"First road edge points length: {len(edge_points)}")
                            except Exception as e:
                                print(f"Error accessing road edge points: {e}")
                    except Exception as e:
                        print(f"Error accessing road edges: {e}")
                print()

                # Check for other fields
                print("Other available fields:")
                for field in dir(model_output):
                    if not field.startswith('_'):
                        try:
                            value = getattr(model_output, field)
                            if hasattr(value, '__len__'):
                                print(f"  {field}: {type(value)}, length: {len(value)}")
                            else:
                                print(f"  {field}: {type(value)}")
                        except:
                            print(f"  {field}: <error accessing>")

                print("\n" + "="*50 + "\n")
                break  # Only process one message for debugging

    except KeyboardInterrupt:
        print("\nStopping debug...")
    except Exception as e:
        print(f"Error in debug: {e}")

if __name__ == "__main__":
    debug_model_output()