float3 tri_normal(float3 t1, float3 t2, float3 t3, float3 D){
    float3 norm = cross(t2-t1, t3-t1);
    float len = sqrt(dot(norm, norm));
    if(!(len > 0 || len < 0)) return norm;
    float3 normalizednorm = norm / len;
    if(dot(D, norm) > 0) return -normalizednorm;
    else return normalizednorm;
}

int ray_sphere_noi(float3 c, float Rsq, float3 o, float3 d){
    float3 v = cross(d, c - o);
    if(dot(v,v) < Rsq) return 1;
    return 0;
}

int ray_box_noi(float3 O, float3 D, float3 minv, float3 maxv, float t0, float t1) {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    if(D.x >= 0){
        tmin = (minv.x - O.x) / D.x;
        tmax = (maxv.x - O.x) / D.x;
    }
    else{
        tmin = (maxv.x - O.x) / D.x;
        tmax = (minv.x - O.x) / D.x;
    }
    if(D.y >= 0){
        tymin = (minv.y - O.y) / D.y;
        tymax = (maxv.y - O.y) / D.y;
    }
    else{
        tymin = (maxv.y - O.y) / D.y;
        tymax = (minv.y - O.y) / D.y;
    }

    if( (tmin > tymax) || (tymin > tmax) ) return 0;

    if(tymin > tmin) tmin = tymin;
    if(tymax < tmax) tmax = tymax;
    if(O.z >= 0){
        tzmin = (minv.z - O.z) / D.z;
        tzmax = (maxv.z - O.z) / D.z;
    }
    else{
        tzmin = (maxv.z - O.z) / D.z;
        tzmax = (minv.z - O.z) / D.z;
    }

    if ( (tmin > tzmax) || (tzmin > tmax) ) return 0;

    if(tzmin > tmin) tmin = tzmin;
    if(tzmax < tmax) tmax = tzmax;

    if( (tmin < t1) && (tmax > t0) ) return 1;
    return 0;
}

int ray_tri(float3 v1, float3 v2, float3 v3, float3 O, float3 D, float3* pi1, float *it){
    float3 e1, e2;
    float3 P, Q, T;
    float det, inv_det, u, v;
    float t;
    float EPSILON = 0.000001f;

    e1 = v2-v1;
    e2 = v3-v1;

    P = cross(D, e2);
    det = dot(e1, P);
    if(det > -EPSILON && det < EPSILON) return 0;
    inv_det = 1.0f / det;

    T = O-v1;
    u = dot(T, P) * inv_det;
    if(u < 0.0f || u > 1.0f) return 0;
    
    Q = cross(T, e1);

    v = dot(D, Q) * inv_det;
    if(v < 0.0f || u+v > 1.0f) return 0;

    t = dot(e2, Q) * inv_det;
    if(t > EPSILON){
        *pi1 = O + D*t;
        *it = t;
        return 1; // t is intersection time
    }
    return 0;
}

void set_camera(float4* O, float4* D, uint x, uint y, int w, int h){
    *O = (float4)(w/2, h/2, -500, 0);
    *D = (float4)(x, y, 0, 0) - (*O);
    *D = -*D;
}

float get_rand(__global float* random, unsigned int* hashed, int rand_len){
    float number = random[*hashed];
    *hashed = (*hashed + 1) % rand_len;
    return number;
}

__kernel void path_trace_simple_noaccel(__global float4* vertices, __global float* random, __global float* outarr, int w, int h, int n_tris, int bx, int by, int rand_len)
{
    unsigned int x = get_global_id(0) + bx;
    unsigned int y = get_global_id(1) + by;

    unsigned int hashed = (x*x * y - (y - x)*(y - x)) % rand_len;

    float4 O, D;
    set_camera(&O, &D, x, y, w, h);
    float4 oO, oD;
    oO = O;
    oD = D;

    float4 t0, t1, t2;
    float3 pi1;
    float3 norm;
    float3 rel;
    float4 inter;
    float4 ptemp;

    int intersected = 0;
    int any_intersection = 0;
    int n_collisions = 0;

    float SCALE = 800.0f;
    float4 TRANSFORM = (float4)(1200.0f, 900.0f, -2000.0f, 0.0f);
    TRANSFORM.x -= 600.0f;

    int never_inter = 0;

    int nearest = 0;
    float sdsq, dsq;
    sdsq = -1.0f;
    float it;

    float R, G, B;
    R = G = B = 0.0f;

    int i;
    int maxbounces = 5;
    int samples = 200;

    float A1, B1;
    float pi = 3.14159265f;

    float3 SC = (float3)(1.0f, 1.0f, 3.0f) * SCALE + TRANSFORM.xyz;
    float SR = 3.0f * SCALE;
    float Rsq = SR*SR;

    for(int sample = 0; sample < samples; sample++){
        O = oO;
        D = oD;
        for(int bounce = 0; bounce < maxbounces; bounce++){
            
            n_collisions = 0;
            any_intersection = 0;

            i = 0;
            while(i < n_tris){
                t0 = vertices[i*3]*SCALE+TRANSFORM;
                t1 = vertices[i*3+1]*SCALE+TRANSFORM;
                t2 = vertices[i*3+2]*SCALE+TRANSFORM;
                
                intersected = ray_tri(t0.xyz, t1.xyz, t2.xyz, O.xyz, D.xyz, &pi1, &it);
                if(intersected == 0){
                    
                }
                else{
                    any_intersection = 1;
                    
                    rel = O.xyz - pi1;
                    dsq = dot(rel, rel);
                    if(it > 0) dsq = it;

                    if(dsq < sdsq || sdsq < 0.0f){ 
                        sdsq = dsq;
                        nearest = i;
                        inter = (float4)(pi1, 0.0f);
                    }

                    n_collisions++;
                }
                i++;
            }

            if(any_intersection == 0){
                if(ray_sphere_noi(SC, Rsq, O.xyz, D.xyz) == 1){
                    if(bounce == 0){
                        never_inter = 1;
                    }
                    else{
                        R += 1.0f;
                        G += 1.0f;
                        B += 1.0f;
                    }
                }
                else if(bounce == 0){
                    never_inter = 2;
                }
                break;
            }
            else{
                i = nearest; 
                O = inter;
                float A1 = get_rand(random, &hashed, rand_len) * pi;
                float B1 = get_rand(random, &hashed, rand_len) * pi * 2;
                ptemp = (float4)(sin(A1)*cos(B1), sin(A1)*sin(B1), cos(A1), 0.0f);
                if(dot(ptemp.xyz, D.xyz) > 0){
                    ptemp = -ptemp;
                }
                D = ptemp;
            }
            if(get_rand(random, &hashed, rand_len) < 0.2f){
                break;
            }
        }
        if(never_inter){
            break;
        }
    }

    if(never_inter == 1){
        outarr[(x*h+y)*3] = 1.0f; 
        outarr[(x*h+y)*3+1] = 1.0f; 
        outarr[(x*h+y)*3+2] = 1.0f; 
    }
    else if(never_inter == 2){
        outarr[(x*h+y)*3] = 0.0f; 
        outarr[(x*h+y)*3+1] = 0.0f; 
        outarr[(x*h+y)*3+2] = 0.0f; 
    }
    else{
        outarr[(x*h+y)*3] = R/((float)samples); 
        outarr[(x*h+y)*3+1] = G/((float)samples); 
        outarr[(x*h+y)*3+2] = B/((float)samples); 
    }
}

__kernel void raytrace_normal_noaccel(__global float4* vertices, __global float* outarr, int w, int h, int n_tris, int bx, int by)
{
    unsigned int x = get_global_id(0) + bx;
    unsigned int y = get_global_id(1) + by;

    float4 O, D;
    set_camera(&O, &D, x, y, w, h);

    float4 t0, t1, t2;
    float3 pi1;
    float3 norm;
    float3 rel;

    int intersected = 0;
    int any_intersection = 0;
    int n_collisions = 0;

    float SCALE = 800.0f;
    float4 TRANSFORM = (float4)(1200.0f, 900.0f, -2000.0f, 0.0f);

    int nearest = 0;
    float sdsq, dsq;
    sdsq = -1.0f;
    float it;

    int i = 0;

    while(i < n_tris){
        t0 = vertices[i*3]*SCALE+TRANSFORM;
        t1 = vertices[i*3+1]*SCALE+TRANSFORM;
        t2 = vertices[i*3+2]*SCALE+TRANSFORM;
        
        intersected = ray_tri(t0.xyz, t1.xyz, t2.xyz, O.xyz, D.xyz, &pi1, &it);
        if(intersected == 0){
            
        }
        else{
            any_intersection = 1;
            
            rel = O.xyz - pi1;
            dsq = dot(rel, rel);
            if(it > 0) dsq = it;

            if(dsq < sdsq || sdsq < 0.0f){ 
                sdsq = dsq;
                nearest = i;
            }

            n_collisions++;
        }
        i++;
    }

    if(any_intersection == 0){
    }
    else{
        i = nearest; 
        t0 = vertices[i*3]*SCALE+TRANSFORM;
        t1 = vertices[i*3+1]*SCALE+TRANSFORM;
        t2 = vertices[i*3+2]*SCALE+TRANSFORM;
        norm = tri_normal(t0.xyz, t1.xyz, t2.xyz, D.xyz); 
        float R, G, B;

        float C = 1.0f - pow(0.9f, (float)(n_collisions));
        

        R = (norm.x+1.0f)/2.0f; 
        G = (norm.y+1.0f)/2.0f; 
        B = (norm.z+1.0f)/2.0f; 

        //R = G = B = C;

        outarr[(x*h+y)*3] = R; 
        outarr[(x*h+y)*3+1] = G; 
        outarr[(x*h+y)*3+2] = B; 
    }
}

__kernel void extremes_min(__global float4* winners, __global float4* new_winners){
    unsigned int gid = get_global_id(0);
    float4 av, bv;
    av = winners[gid*2], bv = winners[gid*2+1];

    if(av.x < bv.x){
        new_winners[gid].x = av.x;
    }
    else{
        new_winners[gid].x = bv.x;
    }

    if(av.y < bv.y){
        new_winners[gid].y = av.y;
    }
    else{
        new_winners[gid].y = bv.y;
    }

    if(av.z < bv.z){
        new_winners[gid].z = av.z;
    }
    else{
        new_winners[gid].z = bv.z;
    }
}

__kernel void extremes_max(__global float4* winners, __global float4* new_winners){
    unsigned int gid = get_global_id(0);
    float4 av, bv;
    av = winners[gid*2], bv = winners[gid*2+1];

    if(av.x > bv.x){
        new_winners[gid].x = av.x;
    }
    else{
        new_winners[gid].x = bv.x;
    }

    if(av.y > bv.y){
        new_winners[gid].y = av.y;
    }
    else{
        new_winners[gid].y = bv.y;
    }

    if(av.z > bv.z){
        new_winners[gid].z = av.z;
    }
    else{
        new_winners[gid].z = bv.z;
    }
}

__kernel void sum_verts(__global float4* winners, __global float4* new_winners){
    unsigned int gid = get_global_id(0);

    float4 av = winners[gid*2];
    float4 bv = winners[gid*2+1];

    new_winners[gid] = av + bv;
}

__kernel void sum(__global int* winners, __global int* new_winners){
    unsigned int gid = get_global_id(0);

    int a = winners[gid*2];
    int b = winners[gid*2+1];

    new_winners[gid] = a + b;
}

// attempting instead to use spacial splits
// BVH2

// vertices/3 repr triangles
// planes/2 repr planes by point and normal
// avgpoints/1 is the average point
// em/1 is the min point
// eM/1 is the max point
// belong_tri_ind_len/2 contains which len and ind to determine which split you belong to
// belong_tri/ contains which slit you belong to
// new_planes/2 and new_vertices/3 are where the output is stored
// new_belong_tri_ind_len/2 contains which len and ind to determine which split you belong to
// new_belong_tri/ contains which slit you belong to

// how do i index 
/*
__kernel void spacial_splits(__global float4* planes){
    unsigned int gid = get_global_id(0);
}

typedef struct{
    float4 v1;
    float4 v2;
    float4 v3;
} tri;

typedef struct{
    float4 m;
    float4 M;
} box;

typedef struct{
    float4 abcd;
    float4 p0;
    float4 p1;
    float4 p2;
    float4 n;
} plane;

typedef struct{
    int c1_ind;
    int c2_ind;
    int ld_ind; 
} leaf;

typedef struct{
    box b;
    plane p;
} leaf_data;
*/
float4 ray_plane_ipoint(float4 RO, float4 RD, float4 plane){
    float t = -(dot(plane.xyz, RO.xyz) + plane.w) / dot(plane.xyz, RD.xyz);
    float4 ipoint = RO + t*RD;
    return ipoint;
}

float point_plane_distance(float4 p, float4 plane){
    float d = dot(p.xyz, plane.xyz) + plane.w;
    return d;
}

float4 get_point_in_plane(float4 plane){ // what if a point does not appear in x = 0, y = 0 (parallel to xz-plane)
    float x, y, z;
    if(plane.z != 0.0f){
        x = 0.0f;
        y = 0.0f;
        z = (plane.w - plane.x * x - plane.y * y) / plane.z;
    }
    else if(plane.y != 0.0f){
        x = 0.0f;
        z = 0.0f;
        y = (plane.w - plane.x * x - plane.z * z) / plane.y;
    }
    else if(plane.x != 0.0f){
        y = 0.0f;
        z = 0.0f;
        x = (plane.w - plane.y * y - plane.z * z) / plane.x;
    }
    float4 point = (float4)(x, y, z, 0.0f);   
    return point;
}

__kernel void determine_splits( // 2nd run is messed up cuz now tri_indexes should be different
    __global float4* vertices, // need to find the plane it belongs to as well
    __global float4* planes, // but determine splits first, so we dont need a HUGE amount of output arrays
    __global float4* mv,
    __global float4* Mv,
    __global int* tree_indexes,

    __global int* tri_indexes,

    __global int* out_vertices_type,
    __global int* out_children,

    int tri_indexes_entry_len){

    unsigned int gid = get_global_id(0);
    unsigned int tindx = gid*3;

    unsigned int cchild = 0;

    unsigned int base_index = gid*tri_indexes_entry_len;
    for(int i = base_index+1; i < base_index+tri_indexes_entry_len; i++){
        cchild = tree_indexes[cchild+tri_indexes[i]];
    } // i think this will get us the current child?

    unsigned int child1_ind, child2_ind, m_ind, M_ind, plane_ind;
    plane_ind = tree_indexes[cchild+4];
    float4 plane = planes[plane_ind];
    //child1_ind = tree_indexes[cchild+0];
    //child2_ind = tree_indexes[cchild+1];

    float A, B, C, D;
    A = plane.x, B = plane.y, C = plane.z, D = plane.w;

    float distance;
    float epsilon = 1.18e-11;
    float4 p;
    int points_in_plane = 0;
    int g1_ind, g2_ind, g3_ind;
    g1_ind = g2_ind = g3_ind = 0;

    float4 g1[3];
    float4 g2[3];
    float4 g3[3];

    for(int i = 0; i < 3; i++){
        p = vertices[tindx+i];
        distance = point_plane_distance(p, plane);
        if(distance < -epsilon){
            g2[g2_ind] = p;
            g2_ind++;
        }
        else if(distance > epsilon){
            g1[g1_ind] = p;
            g1_ind++;
        }
        else{
            points_in_plane++;
            g3[g3_ind] = p;
            g3_ind++;
        }
    }

    
    int into_three, into_two, dont_split;
    into_three = 3;
    into_two = 2;
    dont_split = 1;
    int split_type;
    
    
    if((((g1_ind == 1) && (g2_ind == 2)) || ((g1_ind == 2) && (g2_ind == 1)))){
        split_type = into_three;
    }
    else if(points_in_plane == 1){
        if(!(g1_ind == g2_ind == 1)){
            split_type = dont_split;
        }
        else{
            split_type = into_two;
        }
    }
    else{
        split_type = dont_split;
    }

    out_vertices_type[gid] = split_type;

    if(split_type != dont_split){
        out_children[gid] = 2;
    }
    else{
        float4 tv;
        float4 pn;
        float4 pnorm;

        pnorm = plane;
        float4 v_p2M;
        float4 pinplane;
        float4 Mp;

        pinplane = get_point_in_plane(plane);
        Mp = Mv[M_ind];
        v_p2M = Mp - pinplane; 

        int which_child;
        int upper_child = 1;
        int lower_child = 0;

        if(dot(pnorm.xyz, v_p2M.xyz) < 0){
            pnorm = -pnorm;
        }

        which_child = lower_child;
        for(int i = 0; i < 3; i++){
            tv = vertices[tindx+1] - pinplane;
            if(dot(pnorm.xyz, tv.xyz) > 0){
                which_child = upper_child;
                break;
            }
        }
        out_children[gid] = which_child;
    }
}

__kernel void prefix_sums_sum(__global int* A, __global int* B, __global int* sums, int index){
    unsigned int gid = get_global_id(0);
    int sind = gid*2;
    int a = A[sind];
    int b = A[sind+1];
    B[gid] = a + b;
    sums[gid+index] = a + b;
}

int sgeo(int a1, int r, int n){
    return ((a1 / (1-r)) * (1-pow((float)(r), (float)(n+1))));
}

__kernel void prefix_sums_tree(__global int* sums, __global int* tree){
    unsigned int gid = get_global_id(0);

    unsigned int where = (int)(log2((float)(gid) + 1.0f));
    unsigned int pwhere = where - 1;
    unsigned int S = (int)(sgeo(1, 2, where));
    unsigned int pS = (int)(sgeo(1, 2, pwhere));
    unsigned int aindx = aindx = (S-(gid+1))+pS;
    uint c1 = (gid*2+1) * 3;
    uint c2 = (gid*2+2) * 3;
    uint value = sums[aindx];
    tree[gid*3] = c1;
    tree[gid*3+1] = c2;
    tree[gid*3+2] = value;
}

__kernel void prefix_sum(__global int* tree, __global int* out, int divides, int factor){ // can maybe skip making the "tree"; easy? function to compute c1/c2, and correlate with their values
    unsigned int gid_ = get_global_id(0);
    unsigned int gid;
    if(gid_ < 1){
        out[0] = 0;
        return;
    }
    else{
        gid = gid_ - 1;
    }

    int total = tree[2];
    unsigned int bits = divides;
    unsigned int shifted;
    unsigned int direction;
    int rsum;
    unsigned int c1, c2;

    unsigned int tind = 0;

    for(int i = 0; i < divides; i++){
        shifted = (1 << (bits)) >> (i+1);
        direction = gid & shifted;
        c1 = tree[tind];
        c2 = tree[tind+1];
        rsum = tree[c2+2];
        if(direction == 0){ // left
            tind = c1;
            total -= rsum;
        }
        else{ // right
            tind = c2;
        }
    }

    out[gid+1] = total * factor;
}

__kernel void split_triangles( // also need to update tri_indexes (children?)
    __global float4* vertices, // could "sort" the triangles by buckets here too?
    __global float4* planes, // put another array in?
    __global float4* mv,
    __global float4* Mv,
    __global int* tree_indexes,
    __global int* tri_indexes,
    __global int* vertices_type,
    __global int* cumulative_tri_index,
    __global int* children,
    __global float4* out_vertices,
    __global int* out_tri_indexes,

    int tri_indexes_entry_len

    ){
    unsigned int gid = get_global_id(0);
    unsigned int cid = cumulative_tri_index[gid];
    int vert_type = vertices_type[gid];

    unsigned int tindx = gid*3;

    unsigned int cchild = 0;

    unsigned int base_index = gid*tri_indexes_entry_len;
    for(int i = base_index+1; i < base_index+tri_indexes_entry_len; i++){
        cchild = tree_indexes[cchild+tri_indexes[i]];
    } // i think this will get us the current child?

    unsigned int child1_ind, child2_ind, m_ind, M_ind, plane_ind;
    plane_ind = tree_indexes[cchild+4];
    float4 plane = planes[plane_ind];
    //child1_ind = tree_indexes[cchild+0];
    //child2_ind = tree_indexes[cchild+1];

    float A, B, C, D;
    A = plane.x, B = plane.y, C = plane.z, D = plane.w;

    float distance;
    float epsilon = 1.18e-11;
    float4 p;
    int points_in_plane = 0;
    int g1_ind, g2_ind, g3_ind;
    g1_ind = g2_ind = g3_ind = 0;

    float4 g1[3];
    float4 g2[3];
    float4 g3[3];

    for(int i = 0; i < 3; i++){
        p = vertices[tindx+i];
        distance = point_plane_distance(p, plane);
        if(distance < -epsilon){
            g2[g2_ind] = p;
            g2_ind++;
        }
        else if(distance > epsilon){
            g1[g1_ind] = p;
            g1_ind++;
        }
        else{
            points_in_plane++;
            g3[g3_ind] = p;
            g3_ind++;
        }
    }

    float4 t0, t1, t2;
    t0 = vertices[tindx];
    t1 = vertices[tindx+1];
    t2 = vertices[tindx+2];

    float4 RO, RD;
    float4 inter1, inter2;
    float4 osi, tsi1, tsi2;

    float4 a, b, c;

    float angle1, angle2;
    float4 third, fourth;

    if(vert_type == 1){
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;
        //out_tri_indexes[last_tri_indexes_index] = rel_child_ind;
    }
    else if(vert_type == 2){

        // now split irregular
        tsi1 = g1[0];
        tsi2 = g2[0];
        osi = g3[0];
        RO = tsi1;
        RD = tsi2 - tsi1;
        inter1 = ray_plane_ipoint(RO, RD, plane);

        t0 = osi, t1 = inter1, t2 = tsi1;
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;

        t0 = osi, t1 = inter1, t2 = tsi2;
        out_vertices[cid+3] = t0;
        out_vertices[cid+1+3] = t1;
        out_vertices[cid+2+3] = t2;
    }
    else if (vert_type == 3){

        if(g1_ind == 2){
            tsi1 = g1[0];
            tsi2 = g1[1];
            osi = g2[0];
        }
        else if(g2_ind == 2){
            tsi1 = g2[0];
            tsi2 = g2[1];
            osi = g1[0];
        }

        RO = osi;
        RD = tsi1 - osi;
        inter1 = ray_plane_ipoint(RO, RD, plane);
        RO = osi;
        RD = tsi2 - osi;
        inter2 = ray_plane_ipoint(RO, RD, plane);

        t0 = osi, t1 = inter1, t2 = inter2;
        out_vertices[cid] = t0;
        out_vertices[cid+1] = t1;
        out_vertices[cid+2] = t2;

        a = inter2 - inter1;
        b = tsi1 - inter1;
        c = tsi2 - inter1;
        angle1 = acos(dot(a.xyz, b.xyz) / (sqrt(dot(a.xyz, a.xyz)) * sqrt(dot(b.xyz, b.xyz))));
        angle2 = acos(dot(a.xyz, c.xyz) / (sqrt(dot(a.xyz, a.xyz)) * sqrt(dot(c.xyz, c.xyz))));
        if(angle1 < angle2){
            third = tsi1;
            fourth = tsi2;
        }
        else{
            third = tsi2;
            fourth = tsi1;
        }

        t0 = inter1, t1 = inter2, t2 = third;
        out_vertices[cid+3] = t0;
        out_vertices[cid+1+3] = t1;
        out_vertices[cid+2+3] = t2;

        t0 = third, t1 = fourth, t2 = inter1;
        out_vertices[cid+6] = t0;
        out_vertices[cid+1+6] = t1;
        out_vertices[cid+2+6] = t2;

    }

}

