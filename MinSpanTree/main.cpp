#include <iostream>
#include <set>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>
#include <cassert>
#include <sstream>
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

Vertex* initializeVertex(vector<double> coords = {0}) {
    
    // Initialize vertex, make self the parent, and set rank to one
    Vertex* vertex = new Vertex();
    vertex->parent = vertex; //
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
    
    bool in_euclidean_space = (dimensions == 0);
    
    if (!in_euclidean_space) {
        
        // Random weights -- coordinates in space meaningless
        for (int i = 0; i < size; i++) {
            vertices[i] = initializeVertex();
        }
    }
    
    // Coordinates in space instrumental
    for (int i = 0; i < size; i++) {
        vertices[i] = generateRandomVertex(dimensions);
    }
    
    long edge_count = 0;
    for (int i = 0; i < size; i++) {
        
        for (int j = i + 1; j < size; j++) {
            
            Vertex* u = vertices[i];
            Vertex* v = vertices[j];
            
            // Distance == Edge Weight
            double distance = in_euclidean_space ? generateRandomVal() : calcEuclideanDist(u,v);
            
            // TODO Why not use vertex's push_back method?
            Edge* new_edge = new Edge({u, v, distance});
            edges[edge_count++] = new_edge;
        }
    }
    
    return (Graph){(int) vertices.size(), (int) edges.size(), edges, vertices};
}


// Using a struct for comparison is optimized more at low level
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
void testHardcodedGraph() {
    
    // Hardcoded vertices and edges
    Vertex* A = initializeVertex();
    Vertex* B = initializeVertex();
    Vertex* C = initializeVertex();
    Vertex* D = initializeVertex();
    Vertex* E = initializeVertex();
    Vertex* F = initializeVertex();
    Vertex* G = initializeVertex();
    
    Edge* AB = new Edge({A, B, 7.0});
    Edge* AD = new Edge({A, D, 5.0});
    Edge* BC = new Edge({B, C, 8.0});
    Edge* BD = new Edge({B, D, 9.0});
    Edge* BE = new Edge({B, E, 7.0});
    Edge* CE = new Edge({C, E, 5.0});
    Edge* DE = new Edge({D, E, 15.0});
    Edge* DF = new Edge({D, F, 6.0});
    Edge* EF = new Edge({E, F, 7.0});
    Edge* EG = new Edge({E, G, 9.0});
    Edge* FG = new Edge({F, G, 11.0});
    
    // Hardcoded graph and true and false MSTs
    vector<Vertex*> vertices_t {A,B,C,D,E,F,G};
    vector<Edge*> edges_t {AB, AD, BC, BE, CE, EG, FG, DF, EF, DE, BD};
    Graph G_test {7, 11, edges_t, vertices_t};
    set<Edge*> true_MST {AD, CE, DF, AB, BE, EG};
    set<Edge*> false_MST {AD, CE, DF, AB, BE, EF};
    
    // Test
    auto MST = findMST(G_test);
    set<Edge*> found_MST(MST.begin(), MST.end());
    
    assert(found_MST != false_MST);
    assert(found_MST == true_MST);
}

void testUtilityFunctions() {
    /* 
     TODO Test things like euclidean distance. We know the MST search algo works. Just need to make
     sure its dependencies work, too
     */
    
}


/*
COMMAND LINE INTERFACE
*/
int main(int argc, char** argv){
    
    if (argc != 5) {
        return -1;
    }
    
    vector<int> params;
    
    for (int i = 1; i < argc; i++) {
        
        istringstream char_param (argv[i]);
        int int_param;

        if (char_param >> int_param) {
            params.push_back(int_param);
        } else {
            return -1;
        }
    }
    
    int flag = params[0];
    long size = params[1];
    int trials = params[2];
    int dimensions = params[3];
    
    if (dimensions == 1) {
        return -1;
    }
    
    // TODO Make tests cutomizable
    if (flag == 1) {
        cout << "\nTesting\n";
        testHardcodedGraph();
        cout << "\nMST Working on Hardcoded Graph\n";
        cout << "\nAll Tests Pass\n";
        
        return 0;
    }
    
    // TODO record and aggregate data
    for (int trial = 0; trial < trials; trial++) {
        rand_gen.seed(seed_val);
        auto G = generateGraph(size, dimensions);
        auto MST = findMST(G);
    }

    return 0;
}
