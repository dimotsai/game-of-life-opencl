char get(__global unsigned char *map, int x, int y, int w, int h, int gw, int gh)
{
    if( x < 0 || y < 0 || x >= w || y >= h ) return 0;
    else return map[y * gw + x];
}

__kernel void devGolGenerate(__global unsigned char *map_in, __global unsigned char *map_out, __write_only image2d_t image, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height) return;

    int cell_count = 0;
    int gw = get_global_size(0);
    int gh = get_global_size(1);

    if(get(map_in, x-1, y-1, width, height, gw, gh)) ++cell_count;
    if(get(map_in, x-1, y  , width, height, gw, gh)) ++cell_count;
    if(get(map_in, x-1, y+1, width, height, gw, gh)) ++cell_count;
    if(get(map_in, x  , y-1, width, height, gw, gh)) ++cell_count;
    if(get(map_in, x  , y+1, width, height, gw, gh)) ++cell_count;
    if(get(map_in, x+1, y-1, width, height, gw, gh)) ++cell_count;
    if(get(map_in, x+1, y  , width, height, gw, gh)) ++cell_count;
    if(get(map_in, x+1, y+1, width, height, gw, gh)) ++cell_count;

    if(get(map_in, x, y, width, height, gw, gh) == 1 && (cell_count > 3 || cell_count < 2))
        map_out[ y * gw + x ] = 0;
    else if(get(map_in, x, y, width, height, gw, gh) == 0 && cell_count == 3)
        map_out[ y * gw + x ] = 1;
    else
        map_out[ y * gw + x ] = get(map_in, x, y, width, height, gw, gh);

    write_imagef(image, (int2)(x,y), map_out[ y * gw + x ] ? (1.0) : (0.0));
}
