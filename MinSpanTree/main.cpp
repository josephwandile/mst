#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>
#include "test.h"

using namespace std;
const auto time_seed = static_cast<size_t>(time(0));
const auto clock_seed = static_cast<size_t>(clock());
const size_t pid_seed = hash<thread::id>()(this_thread::get_id());
seed_seq seed_val { time_seed, clock_seed, pid_seed };
mt19937_64 rand_gen;

/*
RANDOMIZATION
*/
double generateRandomVal() {

    return generate_canonical<double, 50>(rand_gen);
}


/*
GRAPH GENERATION
*/
typedef struct Vertex {

    vector<double> coords;
    Vertex* parent;
    int rank;
} Vertex;

typedef struct Edge {

    Vertex* u;
    Vertex* v;
    double distance;
} Edge;

typedef struct Graph {
    int V;
    int E;
    vector<Edge*> edges;
    vector<Vertex*> vertices;
} Graph;

Vertex* initializeVertex(vector<double> coords) {
    
    // Initialize vertex, make self the parent, and set rank to one
    Vertex* vertex = new Vertex();
    vertex->parent = vertex;
    vertex->rank = 1;
    vertex->coords = coords;
    
    return vertex;
}

Vertex* generateRandomVertex(int dimensions) {

    vector<double> coords(dimensions);

    // Initialize coordinates to random vector in given number of dimensions
    for (int i = 0; i < dimensions; i++) {
        coords[i] = (generateRandomVal());
    }

    Vertex* vertex = initializeVertex(coords);
    
    return vertex;
}

double calcEuclideanDist(Vertex* u, Vertex* v) {
    double total = 0;
    for (int i = 0; i < (u->coords).size(); i++)
        total += pow((u->coords[i] - v->coords[i]),2);
    return sqrt(total);
}

Graph generateGraph(long size, int dimensions) {

    vector<Vertex*> vertices(size);
    vector<Edge*> edges(size * (size - 1) / 2);

    // Generate vertices
    for (int i = 0; i < size; i++) {
        vertices[i] = generateRandomVertex(dimensions);
    }

    long edge_count = 0;
    for (int i = 0; i < size; i++) {

        for (int j = i + 1; j < size; j++) {

            Vertex* u = vertices[i];
            Vertex* v = vertices[j];
            double distance = calcEuclideanDist(u,v);
            Edge* new_edge = new Edge({u, v, distance});
            edges[edge_count++] = new_edge;
        }
    }

    return (Graph) {(int) vertices.size(), (int) edges.size(), edges, vertices};
}

// Used as the comparison function in the sorting of edges
struct edgeCompare {
    bool operator() (Edge* e1, Edge* e2) {
        return (e1->distance < e2->distance);
    }
} edgeCompare;


/*
DISJOINT SET OPERATIONS
*/
Vertex* find(Vertex* v){
    if(v->parent != v)
        v->parent = find(v->parent);
    return v->parent;
}

void setUnion(Vertex* v, Vertex* u){
    Vertex* v_root = find(v);
    Vertex* u_root = find(u);
    if (v == u){
        return;
    }

    if (v_root->rank < u_root->rank){
        v_root->parent = u_root;
    }
    else if (v_root->rank > u_root->rank){
        u_root->parent = v_root;
    }
    else {
        u_root->parent = v_root;
        v_root->rank++;
    }

}


/*
KRUSKAL'S MST ALGORITHM
*/
void inline sortGraphEdgeList(Graph& G){
    // TODO Might want to consider a partial sort
    sort(G.edges.begin(), G.edges.end(), edgeCompare);
}

vector<Edge*> findMST(Graph& G){
    vector<Edge*> MST;
    sortGraphEdgeList(G);
    for(Edge* E : G.edges){
        if (find(E->u) != find(E->v)){
            MST.push_back(E);
            setUnion(E->u, E->v);
        }
    }
    return MST;
}


/*
TESTING
*/
// Coordinates irrelevant
vector<double> default_coords = {0};
Vertex* A = initializeVertex(default_coords);
// Hardcode Graph



/*
PROGRAM INTERFACE
*/
int main(int argc, char** argv){
    testing();
    rand_gen.seed(seed_val);
    auto G = generateGraph(4, 2);
    auto MST = findMST(G);
    return 0;
}
