#include <iostream>
#include <set>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>
#include <cassert>
#include <fstream>
#include <sstream>
#include <ctime>
#include <String>

using namespace std;
const auto time_seed = static_cast<size_t>(time(0));
const auto clock_seed = static_cast<size_t>(clock());
const size_t pid_seed = hash<thread::id>()(this_thread::get_id());
seed_seq seed_val { time_seed, clock_seed, pid_seed };
mt19937_64 rand_gen;


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

// Used to generate points in space or edge weights in the '0D' graph
double generateRandomVal() {

    // Random value between 0 and 1
    return generate_canonical<double, 50>(rand_gen);
}

// Vertices == singleton sets in disjoint set data structure
Vertex* initializeVertex(vector<double> coords = {0}) {

    // Initialize vertex, make self the parent, and set rank to one
    Vertex* vertex = new Vertex();
    vertex->parent = vertex; //
    vertex->rank = 1;
    vertex->coords = coords;

    return vertex;
}

// Used in the 2D, 3D and 4D graphs
Vertex* generateRandomVertex(int dimensions) {

    vector<double> coords(dimensions);

    for (int i = 0; i < dimensions; i++) {
        coords[i] = (generateRandomVal());
    }

    Vertex* vertex = initializeVertex(coords);

    return vertex;
}

/*
 The value of *d holds euclidean distance in the case in which we do not throw out the edge.
 It allows us to stop evaluation of distance without evaluating every pair of coordinates if a
 threshold is reached.
 */
bool calcEuclideanDist(Vertex* u, Vertex* v, double *d, double threshold) {
    double total = 0;
    for (int i = 0; i < (u->coords).size(); i++){
        total += pow((u->coords[i] - v->coords[i]),2);
        if (total > (threshold * threshold))
            return false;
    }
    *d = sqrt(total);
    return true;
}

/*
 See writeup for explanation of k(n) and error bound of 0.02, which was determined to be
 greater than all residuals found during the testing of k(n).
 */
double inline calculatePruningThreshold(long n, int dimension=3){

    switch (dimension)
    {
        case 0:
            return 1.1473 * pow((double) n, 0.296) + 0.015;

        case 2:
            return 1.1473 * pow((double) n, 0.296) + 0.015;

        case 3:
            return 1.1473 * pow((double) n, 0.296) + 0.015;

        case 4:
            return 1.1473 * pow((double) n, 0.296) + 0.015;

        default:
            return 1;

    }

}

Graph generateGraph(long size, int dimensions, double weightThresh) {

    vector<Vertex*> vertices(size);

    // TODO this doesn't make sense in euclidean space
    vector<Edge*> edges;
    long num_edges = weightThresh * (size * (size - 1) / 2);
    edges.reserve(num_edges);

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

    //long edge_count = 0;
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {

            Vertex* u = vertices[i];
            Vertex* v = vertices[j];

            // Get double pointer to hold total euclidean distance
            double *distance = new double();

            // Distance == Edge Weight
            if (in_euclidean_space)
                *distance = generateRandomVal();

            // If euclidean distance is under threshold, then *distance is that value. Otherwise, don't add edge.
            else
                if(calcEuclideanDist(u, v, distance, weightThresh)){
                    Edge* new_edge = new Edge({u, v, *distance});
                    //edges[edge_count++] = new_edge;
                    edges.push_back(new_edge);
                }

            // Free the pointer
            free(distance);
        }
    }

    // TODO Why are these cast to ints? Should they be longs? Obviously that would change the Graph struct
    return (Graph){(int) size, (int) num_edges, edges, vertices};
}

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
typedef struct MST {

    vector<Edge*> path;
    double total_weight;
} MST;

void inline sortGraphEdgeList(Graph& G){
    sort(G.edges.begin(), G.edges.end(), edgeCompare);
}

MST findMST(Graph& G){

    MST foundMST;

    sortGraphEdgeList(G);
    for(Edge* E : G.edges){
        if (find(E->u) != find(E->v)){
            foundMST.path.push_back(E);
            //cout << "Added edge number " << foundMST.path.size() << "." << endl;
            foundMST.total_weight += E->distance;
            setUnion(E->u, E->v);
        }
    }
    return foundMST;
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
    vector<Vertex*> vertices_list {A,B,C,D,E,F,G};
    vector<Edge*> edges_list {AB, AD, BC, BE, CE, EG, FG, DF, EF, DE, BD};
    Graph G_test {7, 11, edges_list, vertices_list};
    MST found_MST = findMST(G_test);

    vector<Edge*> true_path {AD, CE, DF, AB, BE, EG};
    double true_weight = 39;
    MST true_MST {true_path, true_weight};

    double false_weight = 40;
    vector<Edge*> false_path {AD, CE, DF, AB, BE, EF};
    MST false_MST {false_path, false_weight};

    // Test
    assert(found_MST.path == true_MST.path && found_MST.total_weight == true_MST.total_weight);
    assert(found_MST.path != false_MST.path && found_MST.total_weight != false_MST.total_weight);

    for (Edge* E : G_test.edges)
        free(E);
    for (Vertex* V : G_test.vertices)
        free(V);

}

void testMaxWeight(int dimensions, string outputLoc, int numTrials, int minNodes, int maxNodes){
    ofstream outputFile(outputLoc);
    for (int i = minNodes; i <= maxNodes; i++){
        cout << "Doing " << numTrials << " trials for i = " << i << endl;
        double avg = 0.0;
        for(int j = 0; j < numTrials; j++){
            auto G = generateGraph(i, dimensions, 1.0);
            auto MST = findMST(G);
            avg += MST.path.back()->distance;

            // deallocate pointers
            for (Edge* E : G.edges)
                free(E);
            for (Vertex* V : G.vertices)
                free(V);
        }
        avg /= numTrials;
        outputFile << i << "\t" << avg << endl;
    }
}

/*
 COMMAND LINE INTERFACE

 Note that exit code of 0 is a success; 1 is an input failure; 2 is some other failure.
 */
int main(int argc, char** argv){
    // TODO add double threshold parameter

    if (argc != 5) {
        return 1;
    }

    vector<int> params;

    for (int i = 1; i < argc; i++) {

        istringstream char_param (argv[i]);
        int int_param;

        if (char_param >> int_param) {
            params.push_back(int_param);
        } else {
            return 1;
        }
    }

    int flag = params[0];
    long size = params[1];
    int trials = params[2];
    int dimensions = params[3];

    if (dimensions == 1) {
        return 1;
    }

    if (flag == 1) {
        cout << "\nTesting\n";
        testHardcodedGraph();
        cout << "\nMST Working on Hardcoded Graph\n";
        cout << "\nAll Tests Pass\n";

        return 0;
    }

    if (flag == 2) {
        testMaxWeight(2, "5_500_500_trials_2D.txt", 100, 5, 500);
        return 0;
    }

    double total_search_time = 0;
    double avg_search_time = 0;

    for (int trial = 0; trial < trials; trial++) {

        rand_gen.seed(seed_val);

        clock_t gen_start_time = clock();
        auto G = generateGraph(size, dimensions, calculatePruningThreshold(size, dimensions));
        double gen_total_time = (clock() - gen_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Graph Generation:    " << gen_total_time << "s" << endl;

        clock_t search_start_time = clock();
        auto MST = findMST(G);
        double search_total_time = (clock() - search_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Trial " << trial + 1 << ":    " << search_total_time << "s" << endl;

        total_search_time += search_total_time;

        for (Edge* E : G.edges)
            free(E);
        for (Vertex* V : G.vertices)
            free(V);
    }

    avg_search_time = total_search_time / trials;

    cout << "Average search time over " << trials << " trials:    " << avg_search_time << "s" << endl;

    return 0;
}
