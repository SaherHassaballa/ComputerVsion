def flag_pixel_to_gps(
    x_pixel, y_pixel,
    image_width, image_height,
    focal_length_mm, sensor_width_mm,
    altitude_m, yaw_deg,
    lat_cam, lon_cam
):
    # Step 1: حساب الـ GSD (Ground Sampling Distance)
    gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width)

    # Step 2: نحسب إزاحة البكسل من مركز الصورة
    dx = (x_pixel - image_width / 2) * gsd
    dy = (y_pixel - image_height / 2) * gsd

    # Step 3: ندوّر الإزاحة حسب زاوية الـ Yaw
    import math
    yaw_rad = math.radians(yaw_deg)
    dx_rot = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
    dy_rot = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

    # Step 4: نحول الإزاحة لإحداثيات GPS
    from geopy.distance import distance
    from geopy import Point

    origin = Point(lat_cam, lon_cam)
    north_point = distance(meters=dy_rot).destination(origin, 0)    # شمال
    final_point = distance(meters=dx_rot).destination(north_point, 90)  # شرق

    return final_point.latitude, final_point.longitude
