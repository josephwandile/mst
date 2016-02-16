#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>


using namespace std;
const auto time_seed = static_cast<size_t>(time(0));
const auto clock_seed = static_cast<size_t>(clock());
const size_t pid_seed = hash<thread::id>()(this_thread::get_id());
seed_seq seed_val { time_seed, clock_seed, pid_seed };
mt19937_64 rand_gen;
// TODO Could have a separate 'subset' struct which just keeps track of rank and parent for every edge in the edges vector

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

struct edgeCompare {
    bool operator() (Edge* e1, Edge* e2) { return (e1->distance < e1->distance);}
} edgeCompare;

typedef struct Graph {
    int V;
    int E;
    vector<Edge*> edges;
    vector<Vertex*> vertices;
} Graph;

double generateRandomVal() {
    
    return generate_canonical<double, 50>(rand_gen);
}

Vertex* generateRandomVertex(int dimensions) {
    
    // initialize vertex, make self the parent, and set rank to one
    Vertex* vertex = new Vertex();
    vertex->parent = vertex;
    vertex->rank = 1;
    
    vector<double> coords(dimensions);
    
    // initialize coordinates to random vector in given number of dimensions
    for (int i = 0; i < dimensions; i++) {
        coords[i] = (generateRandomVal());
    }
    
    vertex->coords = coords;
    
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
    
    // Instantiate all (n choose 2) edges.
    // TODO edges.reserve();
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
    
    // TODO Sort edges. Maybe with custom sort algorithm.
    
    return (Graph) {(int) vertices.size(), (int) edges.size(), edges, vertices};
}

Vertex* find(Vertex* v){
    if(v->parent != v)
        v->parent = find(v->parent);
    return v->parent;
}

void set_union(Vertex* v, Vertex* u){
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

vector<Edge*> findMST(Graph &G){
    vector<Edge*> edgeList;
    sort(G.edges.begin(), G.edges.end());
    for(Edge* E : G.edges){
        if (find(E->u) != find(E->v)){
            edgeList.push_back(E);
            set_union(E->u, E->v);
        }
    }
    return edgeList;
}
int main(){
    rand_gen.seed(seed_val);
    
    // NB for Jozi- "auto" type is automatic typing, ie best guess (closest to dynamic typing as it gets in cpp)
    auto G = generateGraph(4, 4);
    auto MST = findMST(G);
    return 0;
}

// TODO It makes sense to keep track of parents using indices or pointers but not both.
// findSet(vector<Edge*> edges, index) {
//   if (edges[i].parent != &edges[i]) {
//     edges[i].parent = findSet(edges, parents_index)
//   }
// }
