import math

# CHANNEL_9_APPROX = ('angle',
#     ('F3-C3',(-72.1055,0.295595)),
#     ('T3-P3',(-123.35,0.436175)),
#     ('P3-O1',(-155.645, 0.399655)),
#     ('F4-C3',(-28.619,0.29465)),
#     ('T4-P4',(123.71, 0.43374)),
#     ('P4-O2',(156.535,0.399335))
#     )
CHANNEL_9_APPROX = ('angle',
                    ('FLV',(-4.5,2)),
('FRV',(4.5,2)),
('BP',(0,4.5)),
    ('F3-C3',(-2,1.5)),
    ('T3-P3',(-3.5,-1.5)),
    ('P3-O1',(-2, -1.5)),
    ('F4-C4',(2,1.5)),
    ('T4-P4',(3.5, -1.5)),
    ('P4-O2',(2,-1.5))
    )
def get_channelpos(channame, chan_pos_list):
    if chan_pos_list[0] == 'angle':
        return get_channelpos_from_angle(channame, chan_pos_list[1:])
    elif chan_pos_list[0] == 'cartesian':
        channame = channame.lower()
        for name, coords in chan_pos_list[1:]:
            if name.lower() == channame:
                return coords[0], coords[1]
        return None
    else:
        raise ValueError("Unknown first element "
                         "{:s} (should be type of positions)".format(
            chan_pos_list[0]))
def get_channelpos_from_angle(channame, chan_pos_list=CHANNEL_9_APPROX):
    """Return the x/y position of a channel.

    This method calculates the stereographic projection of a channel
    from ``CHANNEL_10_20``, suitable for a scalp plot.

    Parameters
    ----------
    channame : str
        Name of the channel, the search is case insensitive.

    chan_pos_list=CHANNEL_10_20_APPROX,
    interpolation='bilinear'

    Returns
    -------
    x, y : float or None
        The projected point on the plane if the point is known,
        otherwise ``None``

    Examples
    --------

    >>> plot.get_channelpos_from_angle('F3-C3')
    (0.1720792096741632, 0.0)
    >>> # the channels are case insensitive
    >>> plot.get_channelpos_from_angle('T4-P3')
    (0.1720792096741632, 0.0)
    >>> # lookup for an invalid channel
    >>> plot.get_channelpos_from_angle('F4-C3')
    None

    """
    channame = channame.lower()
    for i in chan_pos_list:
        if i[0].lower() == channame:
            # convert the 90/4th angular position into x, y, z
            p = i[1]
            x, y = _convert_2d_angle_to_2d_coord(*p)
            return x, y
    return None


def _convert_2d_angle_to_2d_coord(a,b):
    # convert the 90/4th angular position into x, y, z
    ea, eb = a * (90 / 4), b * (90 / 4)
    ea = ea * math.pi / 180
    eb = eb * math.pi / 180
    x = math.sin(ea) * math.cos(eb)
    y = math.sin(eb)
    z = math.cos(ea) * math.cos(eb)
    # Calculate the stereographic projection.
    # Given a unit sphere with radius ``r = 1`` and center at
    # the origin. Project the point ``p = (x, y, z)`` from the
    # sphere's South pole (0, 0, -1) on a plane on the sphere's
    # North pole (0, 0, 1).
    #
    # The formula is:
    #
    # P' = P * (2r / (r + z))
    #
    # We changed the values to move the point of projection
    # further below the south pole
    mu = 1 / (1.3 + z)
    x *= mu
    y *= mu
    return x, y