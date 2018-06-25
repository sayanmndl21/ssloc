from detect import Detect
from point import Point
from posdata import position
from collections import namedtuple
from scipy import signal

import math
import scipy.optimize as opt
import sklearn.cluster as clst
import pickle #check persistence example
import numpy as np
import pylab
import bisect, time, os


scaling = 1.4 #scale for standard deviation
min =  0.0001 #distance
Re = 1000*6371

threshold = 2 #distance threshold

def dist_from_sound(d_ref, spl_ref, spl_current):
    dist = d_ref * math.pow(10, (spl_ref - spl_current) / float(20))#reference distance and spl, spl to track
    return dist


def dist_from_detection(x, y, node_event):
    lat1 = math.radians(x)
    lon1 = math.radians(y)
    lat2 = math.radians(node_event.x)
    lon2 = math.radians(node_event.y)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        (math.sin(dlat / 2)) ** 2 +
        math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2)) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS * c

    return distance


def normal_distribution(x):
    distribution = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * math.pow(x, 2))#gets normal distribution
    return distribution


def set_node_events_std(node_events):#standard deviation of node events
    if len(node_events) == 0:
        raise ValueError("Node event list Empty")

    max_time = 0
    min_time = node_events[0].get_timestamp()
    for node_event in node_events:
        if node_event.get_timestamp() > max_time:
            max_time = node_event.get_timestamp()
        elif node_event.get_timestamp() < min_time:
            min_time = node_event.get_timestamp()

    for node_event in node_events:
        time_error = 1.0 - ((max_time - node_event.get_timestamp())/(max_time - min_time + 1))
        node_event.set_std(scaling / (node_event.confidence + time_error))


def position_evaluation(x, y, r_ref, l_ref, node_events):
    set_node_events_std(node_events)

    eval =  sum(
        [
            normal_distribution(
                (
                    dist_from_detection(x, y, n) -
                    dist_from_sound(r_ref, l_ref, n.spl)
                ) / n.get_std()
            ) / n.get_std() for n in node_events
        ]
    )

    return eval


def position_probability(x, y, d_ref, spl_ref, node_events):
    return position_evaluation(x, y, d_ref, spl_ref, node_events) / float(len(node_events))


def determine_sound_position_list(d_ref, spl_ref, node_events, **kwargs):

    p_func = lambda v: -1 * position_probability(
        v[0], v[-1],
        d_ref, spl_ref,
        node_events
    )

    max_list = [
        opt.fmin(p_func, ne.get_pos(), full_output=1, **kwargs)
        for ne in node_events
    ]

    max_vals = [
        (Point(x, y), -z) for (x, y), z, _, _, _ in max_list
    ]

    return max_vals


def determine_peaks(opt_vals, label_list):

    max_prob_list = list()
    max_point_list = list()
    for i, (point, prob) in zip(label_list, opt_vals):
        try:
            if max_prob_list[i] < prob:
                max_point_list[i] = point
                max_prob_list[i] = prob
        except IndexError:
            max_point_list.append(point)
            max_prob_list.append(prob)

    ret_list = list()
    for max_point in max_point_list:
        too_close = False
        for ret_point in ret_list:
            if ret_point.dist_to_lat_long(max_point) < MIN_DIST:
                too_close = True
                break
        if not too_close:
            ret_list.append(max_point)

    return ret_list


def determine_sound_locations_instance(r_ref, l_ref, node_events, **kwargs):

    max_vals = determine_sound_position_list(
        r_ref, l_ref,
        node_events,
        **kwargs
    )

    positions = np.array([p.to_list() for p, _ in max_vals])

    af = clustering.AffinityPropagation().fit(positions)

    max_prob_centers = determine_peaks(max_vals, af.labels_)

    prob_list = [
        position_probability(
            p.x, p.y, r_ref, l_ref,
            node_events
        ) for p in max_prob_centers
    ]

    ret_list = [
        position(p, conf)
        for p, conf in zip(max_prob_centers, prob_list)
    ]

    return ret_list


def evaluate_location_list(location_list):

    if location_list == None:
        return 0

    locations_conf = 0
    for location in location_list:
        locations_conf += location.get_confidence()

    return locations_conf


def determine_reference_data(r_ref, l_ref, node_events, **kwargs):

    pos_func = lambda ref: -1 * evaluate_location_list(
        determine_sound_locations_instance(
            ref[0], ref[1],
            node_events,
            **kwargs
        )
    )

    opt_output = opt.fmin(pos_func, [r_ref, l_ref], full_output=1, **kwargs)

    r_opt, l_opt = opt_output[0]

    return r_opt, l_opt


def get_node_distance_lists(r_ref, l_ref, node_events, locations):

    distance_lists = list()

    for location in locations:

        distance_list = list()

        for node_event in node_events:
            actual_distance= distance_from_detection_event(
                location.x,
                location.y,
                node_event
            )

            predicted_distance = distance_from_sound(
                r_ref, l_ref,
                node_event.get_spl()
            )

            distance_list.append(abs(predicted_distance - actual_distance))

        distance_lists.append(distance_list)

    return distance_lists


def associate_node_events(r_ref, l_ref, node_events, locations):

    distance_lists = get_node_distance_lists(
        r_ref, l_ref,
        node_events,
        locations
    )

    association_dict = dict()

    for location_index, distance_list in enumerate(distance_lists):
        for node_index, distance in enumerate(distance_list):

            if distance < DISTANCE_THRESHOLD:
                if not location_index in association_dict.keys():
                    association_dict[locations[location_index]] = list()
                association_dict[locations[location_index]].append(
                    node_events[node_index]
                )

    return association_dict


def determine_sound_locations(r_ref, l_ref, node_events, **kwargs):

    initial_sound_locations = determine_sound_locations_instance(
        r_ref, l_ref,
        node_events,
        **kwargs
    )

    node_event_associations = associate_node_events(
        r_ref, l_ref,
        node_events,
        initial_sound_locations
    )

    location_list = list()

    for event_list in node_event_associations.values():
        r_opt, l_opt = determine_reference_data(
            r_ref, l_ref,
            event_list,
            **kwargs
        )

        location_list += determine_sound_locations_instance(
            r_opt, l_opt,
            event_list,
            **kwargs
        )

    return location_list

def gauss_kern(size, sizey = None):
    #returns normalized 2D gauss kernel
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x,y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g/g.sum()

def blur_image(im, n, ny=None):
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)
