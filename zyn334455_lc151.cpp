#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>
#include <stack>
#include <queue>
#include <sstream>
#include <algorithm>
#include <assert.h>
using namespace std;

// Part 0: Definitions of data structures
 /* Definition for singly-linked list. */
struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};
 
// use a double linked list to maintain the queue of cache
// double list node
struct node {//double linkedlist node
    node* next;
    node* prev;
    int key;
    int val;
    node(int k, int v, node* n, node* p) : key(k), val(v), next(n), prev(p){};  
    node(int k, int v) : key(k), val(v), next(0), prev(0){};  
};
 struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
 /* Definition for undirected graph. */
 struct UndirectedGraphNode {
      int label;
      vector<UndirectedGraphNode *> neighbors;
      UndirectedGraphNode(int x) : label(x) {};
 };
 /**
 /* Definition for singly-linked list with a random pointer. */
 struct RandomListNode {
     int label;
     RandomListNode *next, *random;
     RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
 };

struct TreeLinkNode {
  int val;
  TreeLinkNode *left, *right, *next;
  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};

 
 // Part I: Single Linked List & Double Linked List
 /*********** Remove Duplicates from Sorted List II **************/
 // 这道题第二遍重点做一下。没做明白。
class RDfSL {
public:
    ListNode *deleteDuplicates(ListNode *head) {
        if(head == NULL || head->next == NULL)
            return head;
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* pre = dummy;
        while(pre->next != NULL && pre->next->next != NULL) {
            ListNode* curr = pre->next->next;
            if(curr->val != pre->next->val) {
                pre = pre->next;
            } 
			else {
                while(curr != NULL && curr->val == pre->next->val) curr = curr->next;
                pre->next = curr;
            }
        }
        return dummy->next;
    }
};
 /************* LRU Cache **************/
class LRUCache{
private:
    unordered_map <int, node*>  hm; // int stores the key of the elem.
    node *head;
    node *tail;
    int capacity;
public:
    LRUCache(int capacity) {
        this->capacity = capacity;
        head = new node(0, 0);
        tail = new node(0, 0);
    }
    
    int get(int key) {
        if (hm.find(key) == hm.end()) {
            return -1;
        }
        node *curr = hm[key];
        if (curr == head) return curr->val;
        if (curr == tail) {
            node *tprev = curr->prev;
            tprev->next = 0;
            curr->next = head;
            head->prev = curr;
            head = curr;
            tail = tprev;
            return curr->val;
        }
        curr->prev->next = curr->next;
        curr->next->prev = curr->prev;
        curr->prev = 0;
        head->prev = curr;
        curr->next = head;
        head = curr;
        return curr->val;
    }
    
    void set(int key, int value) {
        // if already exist in the cache, then set it to the head of the list
        if (hm.find(key) != hm.end()) {
            node *curr = hm[key];
            curr->val = value; //diao zha tian, key could be reset with a new value!!!
            if (curr == head) return;
            if (curr == tail) {
                node *tprev = curr->prev;
                tprev->next = 0;
                curr->next = head;
                head->prev = curr;
                head = curr;
                tail = tprev;   
                return;
            }
            curr->prev->next = curr->next;
            curr->next->prev = curr->prev;
            curr->prev = 0;
            head->prev = curr;
            curr->next = head;
            head = curr;            
            return; // the element already exists in cache;
        }
        node *nn = new node(key, value);
        // what should be noticed is that the cache is empty here;
        if (hm.size() == 0) {
            head = nn;
			tail = nn;
            head->next = tail;
            tail->prev = head;
            hm[key] = nn;
            // delete nn;
            return;
        }
        if (hm.size() == 1) {
			if (hm.size() == capacity) goto FULL;
            nn->next = tail;
            tail->prev = nn;
            head = nn;
            hm[key] = nn;
           //  delete nn;
            return;
        }
        if (hm.size() < capacity) {
            hm[key] = nn;
            nn->next = head;
            head->prev = nn;
            head = nn;
           // delete nn;
            return;
        }
FULL:
        if (hm.size() == capacity) { // in this part we should delete the tail if size is full.
            node *dn = tail;
            tail = tail->prev;
			hm.erase(dn->key);
            hm[key] = nn;
            nn->next = head;
            head->prev = nn;
            head = nn;
            return;            
        }
    }
};
/*** Remove Nth Node From End of List ***/
// 注意几点，因为两个指针往前走的循环条件是(p2!= 0)，所以p1和p2指针还是间隔n个距离，等到p2指针移出链表的时候，p1指向的就是所要删除的节点，然后看p1是不是头节点即可
// 然后再查一下是不是这个链表有第N个尾节点
class RemoveNthNodefEL {
public:
    ListNode *removeNthFromEnd(ListNode *head, int n) {
        if (head == 0) return 0;
        ListNode *p1 = head, *p2 = head;
        ListNode *prev = head;
        while (p2 != 0 && n > 0) {
            p2 = p2->next;
            n--;
        }
        if (n > 1) return 0; // means that this list do not have Nth Node! // or return head?
        // p1->next is the node that we want to delete;
        while (p2 != 0) {
            prev = p1;
            p1 = p1->next;
            p2 = p2->next;
        }
        if (p1 == head) {
            head = p1->next;
            p1->next = 0;
            free(p1);
            return head;
        }
        else {
            ListNode *tmp = p1->next;
            prev->next = tmp;
            p1->next = 0;
            free(p1);
            return head;
        }
    }
};
/******** Insertion Sort List ********/
class InsertionSortList {
public:
    ListNode *insertionSortList(ListNode *head) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *p1, *p2, *prev1, *prev2;
        if (head == 0) return 0;
        p1 = head; p2 = head->next; prev1 = dummy; prev2 = head;
        if (p2 == 0) return head;
        while (p2 != 0) {
            p1 = dummy->next; // 注意这块，看底下的注释，这是输入为3->2->1的情况，一轮循环过后head跑到中间去了，所以这块循环开始时p1应该写为dummy->next 
            prev1 = dummy;
            while (p1 != p2 && p1->val <= p2->val) {
                p1 = p1->next;
                prev1 = prev1->next;
            }
            if (p1 != p2) { // which means p1->val > p2->val, should insert
                ListNode *next = p2->next; // 1
                p2->next = 0;
                prev2->next = next; // 3->1
                prev1->next = p2; // dummy->2->3->1
                p2->next = p1;
                p2 = next;
            }
            else {
                p2 = p2->next;
                prev2 = prev2->next;
            }
        }
        return dummy->next;
    }
};
// Part II: Hashmap / HashSet
/*********** Valid Sudoku **************/
// 第二遍写的话再用三个一维空间来试试看
// 注意一下用index当做hash key的思路，为什么下标是10，就是因为需要用1~9的index
class IsValidSudoku {
public:
    bool checkValid(vector<int> &v, int num) {
        if (v[num] != 0) return false;
        v[num] = 1;
        return true;
    }
    bool isValidSudoku(vector<vector<char> > &board) {
        vector<vector<int>> row(9, vector<int>(10, 0));
        vector<vector<int>> col(9, vector<int>(10, 0));
        vector<vector<int>> submat(9, vector<int>(10, 0));
        int i, j;
        for (i = 0; i < 9; i++) {
            for (j = 0; j < 9; j++) {
                if (board[i][j] == '.') continue;
                int sub = (i/3) * 3 + j/3;
                if (!checkValid(row[i], board[i][j] - '0') | !checkValid(col[j], board[i][j] - '0') | !checkValid(submat[sub], board[i][j] - '0')) return false;
            }
        }
        return true;
    }
};

/*********** Clone Graph **************/
class CloneGraph {
public:
    UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node) {
        if (!node) return 0;
        queue<UndirectedGraphNode *> q;
        unordered_map<UndirectedGraphNode *, UndirectedGraphNode *> record; // the key is origin node and value is the copy node.
        // use dfs algorithm to copy the graph.
        UndirectedGraphNode *head = new UndirectedGraphNode(node->label);
        record[node] = head;
        q.push(node);
        while (!q.empty()) {
            UndirectedGraphNode *curr = q.front();
            q.pop();
            UndirectedGraphNode *nn = 0;
            if (record.find(curr) == record.end()) { // undiscovered node
                nn = new UndirectedGraphNode(curr->label);
                record[curr] = nn;
            }
            else nn = record[curr];
            for (auto it = curr->neighbors.begin(); it != curr->neighbors.end();it++) {
                UndirectedGraphNode *nnn = 0;
                if (record.find(*it) == record.end()) {
                    nnn = new UndirectedGraphNode((*it)->label);
                    record[*it] = nnn;
                    q.push(*it);
                }
                else {
                    nnn = record[*it];
                }
                nn->neighbors.push_back(nnn);
            }
        }
        return head;
    }
};

/*********** Copy List with Random Pointer ************/
class CLRP {
public:
    RandomListNode *copyRandomList(RandomListNode *head) {
        unordered_map<RandomListNode *, RandomListNode *> records;
        if (!head) return 0;
        RandomListNode *dummy = new RandomListNode(0);
        RandomListNode *tmp;
        RandomListNode *restmp;
        for (tmp = head, restmp = dummy; tmp != 0; tmp = tmp->next, restmp = restmp->next) {
            if (records.find(tmp) != records.end()) {
                restmp->next = records[tmp];
                continue;
            }
            else {
                RandomListNode *newNode = new RandomListNode(tmp->label);
                records[tmp] = newNode;
                restmp->next = newNode;
            }
        }
        for (tmp = head, restmp = dummy->next; tmp != 0; tmp = tmp->next, restmp = restmp->next) {
            if (!tmp->random) continue;
            else {
                RandomListNode *rantmp = records[tmp->random];
                restmp->random = rantmp;
            }
        }
        return dummy->next;
    }
};

// Part III : Tree & Binary Tree
/*************  Flatten Binary Tree **************/
 // 这道题第二遍重点做一下。没做明白。
/* Solution 1: Iterative Solution : */
class FBT1 {
public:
    void flatten(TreeNode *root) {
        while ( root ) {
            if ( root->left ) {
                TreeNode *ptr = root->left;
                while ( ptr->right ) ptr = ptr->right;
                ptr->right = root->right;
                root->right = root->left;
                root->left = NULL;
            }
                root = root->right;
        }
    }
};
/* Recursive Solution */
// by zhuge
class FBT2 {
public:
    TreeNode* findRightBottomNode(TreeNode *r) {
        while(r->right != NULL)
            r = r->right;
        return r;
    }
    
    void flatten(TreeNode *root) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        if(root == NULL)
            return;
        if(root->left != NULL && root->right != NULL) {
            flatten(root->left);
            flatten(root->right);
            TreeNode *rb = findRightBottomNode(root->left);
            rb->right = root->right;
            root->right = root->left;
            root->left = NULL;
        } else if(root->left == NULL && root->right != NULL) {
            flatten(root->right);
        } else if(root->left != NULL && root->right == NULL) {
            flatten(root->left);
            root->right = root->left;
            root->left = NULL;
        } else{//root has no children
            return;
        }
    }
};

/*********** Balanced Binary Tree **************/
class BalancedBinaryTree {
public:
    int maxDepth(TreeNode *root) {
        if (!root) return 0;
        int maxLeft = 0, maxRight = 0;
        if (root->left) maxLeft = maxDepth(root->left);
        if (root->right) maxRight = maxDepth(root->right);
        return 1 + max(maxLeft, maxRight);
    }
    bool isBalanced(TreeNode *root) {
        int left = 0, right = 0;
        if (!root) return true;
        if (root->left) left = maxDepth(root->left);
        if (root->right) right = maxDepth(root->right);
        if (abs(left - right) > 1) return false;
    //    isBalanced(root->left); // check left-tree in a recursive fashion
      //  isBalanced(root->right); //
        return isBalanced(root->left) && isBalanced(root->right);
    }
};

/*********** Symmetric Tree **************/
class SymmetricTree {
public:
    bool helper (TreeNode *a, TreeNode *b) {
        if (!a && !b) return true;
        if (!a && b) return false;
        if (a && !b) return false;
        if (a->val != b->val) return false;
        return helper(a->left, b->right) && helper(a->right, b->left);
    }
    bool isSymmetric(TreeNode *root) {
        if (!root) return true;
        return helper(root->left, root->right);
    }
};

/*********** Binary Tree Preorder Traversal   **************/
// recursive is trival, using a stack to implment iterative method 
// 前序遍历的思路是把右节点先存到栈上，如果有左节点那就将curr放到左节点上
class BTreePreTravel {
public:
    vector<int> preorderTraversal(TreeNode *root) {
        stack<TreeNode *> s;
        vector<int> result;
        if (!root) return result;
        TreeNode *curr = root;
        while (curr) {
            result.push_back(curr->val);
            if (curr->right) s.push(curr->right);
            if (curr->left) curr = curr->left;
            else if (!s.empty()) {
                curr = s.top();
                s.pop();
            }
            else curr = 0;
        }
        return result;
    }
};
/*********** Binary Tree Inorder Traversal   **************/
// recursive is trival, using a stack to implment iterative method 
class BTreeInTravel {
public:
    vector<int> inorderTraversal(TreeNode *root) {
        stack<TreeNode *> ss;
        vector<int> res;
        TreeNode *curr;
        if (!root) return res;
        curr = root;
        while (curr) {
            ss.push(curr);
            curr = curr->left;
        }
        while(!ss.empty()) {
            curr = ss.top();
            ss.pop();
            res.push_back(curr->val);
            if (curr->right) {
                curr = curr->right;
                while (curr) {
                    ss.push(curr);
                    curr = curr->left;
                }
            }
        }
        return res;
    }
};
/*********** 一个中序遍历的错误解法1 **************/
class BTreeInTravelW1 {
public:
    vector<int> inorderTraversal(TreeNode *root) {
        stack<TreeNode *> ss;
        vector<int> res;
        TreeNode *curr;
        if (!root) return res;
        ss.push(root);
        curr = root;
        while(curr) {
            if (curr->left) {
                ss.push(curr);
                curr = curr->left; // 问题就在于把叶子节点处理好后想处理上一层时，这句话又搞回到下一层去了，造成死循环
                continue;
            }
            else {
                res.push_back(curr->val);
                if (curr->right) {
                    curr = curr->right;
                    continue;
                }
                else if (!ss.empty()) {
                    curr = ss.top();
                    ss.pop();
                    continue;
                }
                else curr = 0;
            }
        }
        return res;
    }
};
/*********** Binary Tree Postorder Traversal   **************/
// recursive is trival, using a stack to implment iterative method 
// 这个算法没问题，不过真的有必要用一个set来记录集合？有没有更好的方法？
class BTreePostTravel {
public:
    vector<int> postorderTraversal(TreeNode *root) {
        unordered_set<TreeNode *> visited;
        stack<TreeNode *> ss;
        vector<int> res;
        TreeNode * curr;
        if (!root) return res;
        ss.push(root);
        while (!ss.empty()) {
            curr = ss.top();
            if ((curr->right && curr->left && (visited.find(curr->left) != visited.end()) && (visited.find(curr->right) != visited.end())) || ((!curr->right && curr->left && (visited.find(curr->left)) != visited.end())) || (curr->right && !curr->left && (visited.find(curr->right) != visited.end())) || (!curr->right && !curr->left)) 
            { 
				ss.pop();
                res.push_back(curr->val);
                visited.insert(curr);
            }
            else {
                if (curr->right && visited.find(curr->right) == visited.end()) ss.push(curr->right);
                if (curr->left && visited.find(curr->left) == visited.end()) ss.push(curr->left);
            }
        }
        return res;
    }
};
/*********** Minimum Depth of Binary Tree  **************/
// WRONG SOLUTION: get the minimal of each path
class MDBT1 {
public:
	int getMinDepth (TreeNode *root) {
		if (!root) return 0;
		else return 1 + min(getMinDepth(root->left), getMinDepth(root->right));
	}
	int minDepth(TreeNode *root) {
		if (!root) return 0;
		if (!root->left && !root->right) return 1;
		if (!root->left && root->right) return 1 + getMinDepth(root->right);
		if (root->left && !root->right) return 1 + getMinDepth(root->left);
		else return getMinDepth(root);
	}
};
// 上述解法不对的地方是，在返回最小值时，返回的最小值的路径不一定是叶节点的路径
// 在递归的时候，一定要仔细考虑return的base case和return的其他条件，这是递归是否正确的一个基本要素
// CORRECT SOLUTION: recursive manner
class MDBT2 {
public:
 int minDepth(TreeNode *root) {
    if (NULL == root) return 0;
    if (NULL == root->left && NULL == root->right) {
        return 1;
    }
    int minDepthTree = INT_MAX;
    if (NULL != root->left) {
        minDepthTree = min(minDepthTree, minDepth(root->left) + 1);
    }
    if (NULL != root->right) {
        minDepthTree = min(minDepthTree, minDepth(root->right) + 1);
    }

    return minDepthTree;
}
};
// 上述解法的正确性在于， 只有同时不存在左树和右树时，才return 1,也就是最初的叶节点（或者根节点）。
// 只要有左树或者右树存在，则此节点必不是叶节点，因此也就不能return，需要继续count

/***Validate Binary Search Tree***/
class ValidBST {
public:
    bool helper(TreeNode *root, int min, int max) {
        if (!root) return true;
        return root->val > min && root->val < max && helper(root->left, min, root->val) && helper(root->right, root->val, max);
    }
    bool isValidBST(TreeNode *root) {
        helper(root, INT_MIN, INT_MAX);
    }
};

/*** Binary Tree Zigzag Level Order Traversal ***/
class BTZLOT {
public:
    vector<vector<int> > zigzagLevelOrder(TreeNode *root) {
        vector<vector<int>> res;
        if (!root) return res;
        bool parity_flag = true;
        queue<TreeNode *> currLevelNode;
        stack<TreeNode *> nextLevelNode;
        currLevelNode.push(root);
        vector<int> tmp_res;
        while (!currLevelNode.empty()) {
            TreeNode *curr_node = currLevelNode.front();
            currLevelNode.pop();
            tmp_res.push_back(curr_node->val);
            if (parity_flag) {
                if (curr_node->left) nextLevelNode.push(curr_node->left);
                if (curr_node->right) nextLevelNode.push(curr_node->right);
            }
            else {
                if (curr_node->right) nextLevelNode.push(curr_node->right);
                if (curr_node->left) nextLevelNode.push(curr_node->left);
            }
            if (currLevelNode.empty()) { // what has popped is the last node of current level:
                res.push_back(tmp_res);
                tmp_res.clear();
                while (!nextLevelNode.empty()) {
                    currLevelNode.push(nextLevelNode.top());
                    nextLevelNode.pop();
                }
                parity_flag = !parity_flag;
            }
        }
        return res;
    }
};
/*** Populating Next Right Pointers in Each Node II ***/
 // 层序遍历思想
class Solution {
public:
    void connect(TreeLinkNode *root) {
        queue<TreeLinkNode *> curr_level;
        queue<TreeLinkNode *> next_level;
        if (!root) return ;
        curr_level.push(root);
        while (!curr_level.empty()) {
            TreeLinkNode *curr = curr_level.front();
            curr_level.pop();
            TreeLinkNode *l = curr->left;
            TreeLinkNode *r = curr->right;
            if (l) next_level.push(l);
            if (r) next_level.push(r);
            if (!curr_level.empty()) {
                curr->next = curr_level.front();
                continue;
            }
            else {
                curr_level = next_level;
				next_level = queue<TreeLinkNode *> ();
                continue;
            }
        }
    }
};
/*** Construct Binary Tree from Inorder and Postorder Traversal ***/
// 还是递归解法，思路就是在中序把根节点找出来，然后它左边的都是左子树的节点值，它右边都是右子数的节点值，无论怎么遍历，中序和后序的子树的长度和元素集合都是一样的，所以定义四个参数，中序遍历的起始位置，终止位置，和后序遍历的起始位置，终止位置，分别找到左子树和右子树的这四个参数，递归就好。注意终止条件，以及虽然这两个序都是先遍历左子数，但是由于他们的起始位置不一样，所以不能直接将中序的前两个参数放到后序的那两个参数上，不过因为长度必然是一样的，所以直接pend = p_start + length就好。另外，记得传vector的时候一定要传引用，不然递归的话会copy好多vector出来（因为不传引用的话就是值传递，这样势必会复制），导致MLE然后过不了。PreOrder Inorder的思路是一样的，不再重复做了。
class ConsructBtree {
public:
    int findIndex(const vector<int> &inorder, int target) {
        int i;
        for (i = 0; i < inorder.size(); i++) {
            if (inorder[i] == target) return i;
        }
        return -1;
    }
    // 我了个去，记得vector传应用，不然copy vector会浪费许多内存，导致MLE过不了
    TreeNode *helper(const vector<int> &inorder, const vector<int> &postorder, int i_start, int i_end, int p_start, int p_end) {
        // this is the leaf node
        if (p_start == p_end) {
            TreeNode *n = new TreeNode(postorder[p_end]);
            return n;
        }
        int root = postorder[p_end];
        int root_in_index = findIndex(inorder, root);
        bool lt, rt;
        // determine whether root has left subtree & right subtree
        lt = (root_in_index == i_start)? false: true;
        rt = (root_in_index == i_end)? false: true;
        TreeNode *node, *left, *right;
        node = new TreeNode(root);
        if (lt) {
            int left_end = root_in_index - 1;
            int length = left_end - i_start;
            left = helper(inorder, postorder, i_start, left_end, p_start, p_start + length);
            node -> left = left;
        }
        if (rt) {
            int right_start = root_in_index + 1;
            int length = i_end - right_start;
            right = helper(inorder, postorder, right_start, i_end, p_end- 1 - length, p_end -1);
            node -> right = right;
        }
        return node;
    }
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
        int n = inorder.size();
        if (n == 0) return 0;
        TreeNode *res = helper(inorder, postorder, 0, n-1, 0, n-1);
        return res;
    }
};
// Part IV: Binary Search
/*********** Sqrt(x) ***********/
// 注意三个点：
// 1. 用1 - INT_MAX去二分搜索，不然会溢出
// 2. 所以mid * mid有溢出的问题，应用x/mid
// 3. 如果没有整数平方根的话应该取最小的最近的整数，所以返回rightPtr, 且循环条件应设置 <=, 考虑x  = 2的情况，放到最后循环是 left = 2, right = 2，这时候循环break掉，返回的就是2，不符合题意。因此left等于right的时候应该继续循环，让右指针再减一，最后返回右指针就好。注意这一点。对所有二分查找好像都是通用的。
class SqrtX {
public:
    int sqrt(int x) {
        if (x == 0) return 0; // error;
        if (x == 1) return 1;
        int leftPtr = 0;
        int rightPtr = INT_MAX;
        int middle;
        while (leftPtr <= rightPtr) {
            middle = (leftPtr+rightPtr)/2;
            if (middle == x/middle) return middle;
            else if (middle < x/middle) {
                leftPtr = middle + 1;
            }
            else {
                rightPtr = middle - 1;
            }
        }
        return rightPtr;
    }
};
/*********** Search a 2D Matrix ***********/
class SearchMatrix {
public:
    bool searchMatrix(vector<vector<int> > &matrix, int target) {
        int left = 0;
        int right = matrix.size() - 1;
        int mid = (left+right)/2;
        while(left <= right && matrix[mid][0] != target) {
            mid = (left+right)/2;
            if (target < matrix[mid][0]) {
                right = mid - 1;
            }
            else left = mid + 1;
        }
        if (matrix[mid][0] == target) return true;
        int left_col = 0;
        right = right>=matrix.size()?mid:right;
        int right_col = matrix[right].size() - 1;
        int mid_col = (left_col + right_col) /2;
        while (left_col <= right_col) {
            mid_col = (left_col + right_col) /2;
            if (matrix[right][mid_col] == target) return true;
            else if(matrix[right][mid_col] < target) {
                left_col = mid_col + 1;
            }
            else right_col = mid_col -1;
        }
        return false;
    }
};

// Part V: Bit Manipulation
/*** Single Number II ***/
// bit manipulation还不是很熟练，回头再写一遍
class SN2 {
public:
    int singleNumber(int A[], int n) {
		int N = 32; // 1int -- 4byte * 8bits/byte = 32bits; so N = 32 bits for each integer;
        vector<int> bit_records(N, 0);
        int i,j;
        for (i = 0; i < n; i++) {
            int tmp = A[i];
            for (j = 0; j < N; j++) {
                bit_records[N - j - 1] += tmp & 1; // to check the least significant bit
                tmp = tmp >> 1; // remove the least significant bit
            }
        }
        int res = 0;
        for (i = 0; i < N; i++) {
            res = res << 1;
            if (bit_records[i] %3 == 1) res += 1;
        }
        return res;
    }
};
/*** Single Number ***/
class SN {
public:
    int singleNumber(int A[], int n) {
        int i;
        int tmp = A[0];
        if (n == 1) return tmp;
        for (i = 1; i < n; i++) {
            tmp = tmp ^ A[i]; 
        }
        return tmp;
    }
};
/******** Gray Code *********/
// Gray Code的规律是：n位的gray code可以由n-1位的gray code + n-1位的gray code的倒排的每个元素加1<<n-1即可;
// 这个递归的算法复杂度就是O(n)，没有重复计算（个人觉得是这样）
class GrayCode {
public:
    vector<int> grayCode(int n) {
        vector<int> res;
        if (n == 0) {
            res.push_back(0);
            return res;
        }
        if (n == 1) {
            res.push_back(0);
            res.push_back(1);
            return res;
        }
        int factor = 1<<n-1;
        vector<int> prev = grayCode(n-1);
        res = prev;
        for (int i = prev.size()-1; i>= 0; i--) {
            res.push_back(prev[i] + factor);
        }
        return res;
    }
};

// Part VI: Dynamic Programming
// 动态规划本质上实际就是当前层的问题可以利用过去层的答案来解决，然后把过去层答案记下来，时间复杂度就降低了，本质上就是干了这么一个事。所以问题一定是可递推的就是了。

/*********** Palindrome Partitioning I  **************/
// 思路：动态规划，用两个指针i, j，外循环i从1~n-1, 内循环j从0~i。每次判断子串s.substr(j, i)是不是回文的，如果是的话找到三维数组record[j-1]（也就是从0-j子串的partition的）的所有结果，遍历一遍把substr(j,i)push进去，由此就得到了当前循环i的partition的所有结果。最后i做完n-1后直接返回record[n-1]就是整个字符串的partition。这个算法的时间复杂度O(n^3), 空间复杂度O(n^3)
// 需要注意的一点是不要用records[i] = ...这种去赋值，会报越界问题。除非之前已经进行了初始化，size不为0，不然vector的size是0，然后再record[0]这种，就是越界了，因为用record[0]的话size必然大于等于1
class PalinPartI {
private:
    bool isPalindrome(string s) {
        int n = s.size();
        int p1 = 0, p2 = n - 1;
        while (p1 < n && p2 >= 0 && p1 <= p2) {
            if (s[p1] != s[p2]) return false;
            else {
                p1++;
                p2--;
            }
        }
        return true;
    }
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        int n = s.size();
        vector<vector<vector<string>>> records;
        records.reserve(n);
        if (n == 0) return res;
        if (n == 1) {
            vector<string> t;
            t.push_back(s);
            res.push_back(t);
            return res;
        }
        vector<string> t;
        t.push_back(s.substr(0,1));
        vector<vector<string>> temp_res;
        temp_res.push_back(t);
		records.push_back(temp_res);
		// records.at(0) = temp_res; 注意这句话，第一次在oj写的时候这块出了runtime error，原因就是vector的size还是0然后用了vector[0], 所以除非是已经用vector(n, class<T>) 进行了初始化，vector已经有了size, 否则一律用push_back赋值
        int i, j;
        for (i = 1; i < n; i++) {
            temp_res.clear();
            for (j = 0; j <= i; j++) {
                if (isPalindrome(s.substr(j, i-j+1))) {
                    if (j > 0) {
                        for (auto it = records[j-1].begin(); it != records[j-1].end(); it++) {
                            vector<string> curr = *it;
                            curr.push_back(s.substr(j,i-j+1));
                            temp_res.push_back(curr);
                        }
                    }
                    else {
                        vector<string> t;
                        t.push_back(s.substr(j, i-j+1));
                        temp_res.push_back(t);
                    }
                }
            }
			records.push_back(temp_res);
        }
        res = records[n-1];
        return res;
    }
};
/*********** Palindrome Partitioning II **************/
// 这个版本结果是对的，但是OJ是判为Memory Limited Exception了
// 而且好像没有用到真正用到动态规划? 最好是用一维数组
class PalinPartII {
public:
	bool judgePalindrome (string s, int pt1, int pt2) {
		if (pt1 >= pt2) return true;
		if (s[pt1] != s[pt2]) return false;
		return judgePalindrome(s, pt1+1, pt2-1);
	}
    int minCut(string s) {
		int n = s.size();
		vector<vector<bool>> records(n, vector<bool>(n, false)); //represents there is a palindrome between string[j]-[i]
		for (int a = 0; a < n; a++) {
			records[a][a] = true;
		}
		int i,j;
		for (i = 0; i < n; i++) {
			for (j = 0; j <=i; j++) {
				if (judgePalindrome(s, j, i)) {records[i][j] = true; break;}
			}
		}
		i = n-1;
		int cut = 0;
		while (i>= 0) {
			for (j = 0; j < i; j++) {
				if (records[i][j] == true) {
					if (j == 0) return cut;
					else break;
				} 
			}
			cut++;
			if (j != i) i = j;
			else i--;
		}
		return cut;
    }
};
/*********** Edit Distance **************/
//s1 has length m, s2 has length n
// string s1 = "a", string s2 "ab"
// 所以说我们需要把 m弄成n
// s1 = "green", s2 = "red"
// subsititution: A[i][j] = A[i-1][j-1] + [0/1];
// deletion: A[i][j] = A[i-1][j] e.g, "green" -> "red": deleting 'n' from results of "gree" -> "red"
// insertion A[i][j] = A[i][j-1] e.g, "gre" -> "red" : inserting 'd' from results of "gre" -> "re" 
// 这里的results是指从"A"变到"B"的一系列动作。
// 算是有点想明白了，再好好体会体会！
// 转移矩阵(s1 = "green", s2 = "red")
// A =
//   1     2     3
//   1     2     3
//   2     1     2
//   3     2     2
//   4     3     3
class EditDisance {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();
        if (m == 0) return n; // n insertions
        if (n == 0) return m; // m deletions
        bool m_fill = false, n_fill = false;
        vector<vector<int>> records(m, vector<int>(n, 0)); // Initialization
        if (word1[0] == word2[0]) {
            records[0][0] = 0;
            n_fill = true;
            m_fill = true;
        }
        else records[0][0] = 1; // base case
        int subsititution, deletion, insertion;
        int i, j;
        for (j = 1; j < n; j++) {
            if (!n_fill && (word1[0] == word2[j])) {
                records[0][j] = records[0][j-1];
                n_fill = true;
            }
            else records[0][j] = records[0][j-1] + 1;
        }
        for (i = 1; i < m; i++) {
            if (!m_fill && (word1[i] == word2[0])) {
                records[i][0] = records[i-1][0];
                m_fill = true;
            }
            else records[i][0] = records[i-1][0] + 1;
        }
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                subsititution = (word1[i] == word2[j])?records[i-1][j-1]:records[i-1][j-1]+1;
                deletion = records[i-1][j] + 1;
                insertion = records[i][j-1] + 1;
                records[i][j] = (subsititution<min(deletion, insertion))?subsititution:((deletion<insertion)?deletion:insertion);
            }
        }
        return records[m-1][n-1];
    }
};
// Sub - Topic of DP: 局部解最优解
/*********** Best Time to Buy and Sell Stock  **************/
/*
这是一道非常经典的动态规划的题目，用到的思路我们在别的动态规划题目中也很常用，以后我们称为”局部最优和全局最优解法“。
基本思路是这样的，在每一步，我们维护两个变量，一个是全局最优，就是到当前元素为止最优的解是，一个是局部最优，就是必须包含当前元素的最优的解。接下来说说动态规划的递推式（这是动态规划最重要的步骤，递归式出来了，基本上代码框架也就出来了）。假设我们已知第i步的global[i]（全局最优）和local[i]（局部最优），那么第i+1步的表达式是：
local[i+1]=Math.max(A[i], local[i]+A[i])，就是局部最优是一定要包含当前元素，所以不然就是上一步的局部最优local[i]+当前元素A[i]（因为local[i]一定包含第i个元素，所以不违反条件），但是如果local[i]是负的，那么加上他就不如不需要的，所以不然就是直接用A[i]；
global[i+1]=Math(local[i+1],global[i])，有了当前一步的局部最优，那么全局最优就是当前的局部最优或者还是原来的全局最优（所有情况都会被涵盖进来，因为最优的解如果不包含当前元素，那么前面会被维护在全局最优里面，如果包含当前元素，那么就是这个局部最优）。

接下来我们分析一下复杂度，时间上只需要扫描一次数组，所以时间复杂度是O(n)。空间上我们可以看出表达式中只需要用到上一步local[i]和global[i]就可以得到下一步的结果，所以我们在实现中可以用一个变量来迭代这个结果，不需要是一个数组，也就是如程序中实现的那样，所以空间复杂度是两个变量（local和global），即O(2)=O(1)。
*/
class MaxProfit_I {
public:
    int maxProfit(vector<int> &prices) {
        int local = 0;
        int global = 0;
        for (int i = 1; i < prices.size(); i++) {
            local = max(0, local + prices[i] - prices[i-1]);
            global = max(global, local);
        }
        return global;
    }
};
/*********** Maximum Product Subarray  **************/
// 这个题自己有思路，但是写的时候有些乱，有bug，回头再写一遍
// 底下的思路需要仔细体会
class MaxProduct {
public:
int maxProduct(int A[], int n) {
    if(n==1) return A[0];
    int pMax=0, nMax=0, m = 0;
    for(int i=0; i<n; i++){
        if(A[i]<0) swap(pMax, nMax);
        pMax = max(pMax*A[i], A[i]);
        nMax = min(nMax*A[i], A[i]);
        m = max(m, pMax);
    }
    return m;
}
};
/*********** Longest Substring Without Repeating Characters  **************/
// 思路：维护一个局部解，是当前字母为最右端时的最长不重复子串，然后每次跟global去比较。
class LengthOfLongestSubstring {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> records; // postion of traversed characters
        int temp = 0, global = 0;
        int i;
        int n = s.size();
        if (n == 0) return 0;
        if (n == 1) return 1;
        for (i = 0; i < n; i++) {
            if (records.find(s[i]) != records.end()) {
                auto it = records.find(s[i]);
                int old_pos = it->second;
				for (auto it = records.begin(); it != records.end(); ) { // 记得这块是怎么删除的。
					if (it->second < old_pos) {
						auto temp = it;
						it++;
						records.erase(temp);
						continue;
					}
					it++;
				}
                temp = i - old_pos;
                records[s[i]] = i;
            }
            else {
                records[s[i]] = i;
                temp = temp + 1;
            }
            global = max (global, temp);
        }
        return global;
    }
};
// Sub - Topic: 一维动态规划
/*********** Climbing Stairs  **************/
class ClimbStairs {
public:
    int climbStairs(int n) {
        int first = 1, second = 2, i, result = 0; // first means the first item f(n) = f(n-1) + f(n-2)
        if (n == 1) return first;
        if (n == 2) return second;
        for (i = 3; i <= n; i++) {
            result = first + second;
            first = second;
            second = result;
        }
        return result;
    }
};
/*********** Decode Ways ************/
class DW {
public:
    int numDecodings(string s) {
        int res;
        int n = s.size();
        if (n == 0) return 0;
        if (s[0] == '0') return 0;
        if (n == 1) {
            return s[0]=='0'?0:1;
        }
        vector<int> records(n, 0);
        records[0] = 1;
        if (s[1] - '0' == 0) { 
            if (s[0] != '1' && s[0] != '2') return 0; // wrong input
            else records[1] = 1;
        }
        else if(s[0] -'0' == 1) records[1] = 2;
        else if(s[1] -'0' > 0 && s[1] - '0' < 7 && s[0] - '0' == 2) records[1] = 2;
        else records[1] = 1;

        if (n == 2) return records[1];
        for (int i = 2; i < n; i++) {
            if (s[i] - '0' == 0) {
                if (s[i-1] - '0' == 1 || s[i-1] - '0' == 2) records[i] = records[i-2];
                else return 0;
            }
            else if (s[i-1] == '0') records[i] = records[i-1];
            else if (s[i-1] - '0' == 1) records[i] = records[i-2] + records[i-1];
            else if (s[i-1] -'0' == 2 && s[1] -'0' > 0 && s[i] - '0' < 7 ) records[i] = records[i-2] + records[i-1];
            else records[i] = records[i-1];
        }
        return records[n-1];
    }
};
// Greedy Algorithm
/*********** Container With Most Water  ************/
// 设两个指针，如果左指针小于右指针，移动右指针只会让面积变小，因为瓶颈在左指针，所以当左指针小于右指针时我们移动左指针。反之我们移动右指针
class MaxArea {
public:
    int maxArea(vector<int> &height) {
		int i, j;
		int res = 0;
		int n = height.size();
		if (n == 0 || n == 1) return 0;
		i = 0;
		j = n - 1;
		while (i < j) {
		    res = max(res, min(height[i], height[j]) * (j-i));
		    if (height[i] < height[j]) i++;
		    else j--;
		}
		return res;
    }
};
/*********** Word Break II ***********/
// 注意这个题目的意思，如果划分的任意一段不是字典词，那么这个划分就不算数的
// 思路：动态规划 + NP-Problem。用一个二维布尔变量记录j - i是不是字典词。然后用一个递归的breakHelper从最右段向左查找字典词。如果j -i 是字典词，那么j就是个delimiter, 然后从j-1往下找。注意递归的base case是-1。targetIndex是入口
// 注意delimiter，如果是当前层没找到就应该是pop_back, 而不是清空它，因为高层targetIndex上还放着delimiter，当前循环没法将delimiter递归到0，下一层循环还可以继续递归看是不是可以将delimiter递归到0
class WordBreakII {
private:
    void breakHelper(vector<int> &delimiter, vector<string> &res, const string &s, const vector<vector<bool>> &records, int targetIndex) {
        int j;
        if (targetIndex == -1) {
            string space(" ");
            string tmp = s;
            for (auto it = delimiter.begin(); it != delimiter.end(); it++) {
                int pos = *it;
                if ( pos != 0 ) tmp.insert(pos, space);
            }
            res.push_back(tmp);
            delimiter.pop_back();
            return;
        }
        for (j = targetIndex; j >= 0; j--) {
            if (records[targetIndex][j]) {
                delimiter.push_back(j);
                breakHelper(delimiter, res, s, records, j-1);
                // if (delimiter.empty()) continue;
            }
        }
        delimiter.pop_back();
        return;
    }
    
public:
    vector<string> wordBreak(string s, unordered_set<string> &dict) {
        int n = s.size();
        vector<vector<bool>> records(n, vector<bool>(n, false)); // in the i loop, j would be the index of delimiter
        vector<string> res;
        if (n == 0) return res;
        if (n == 1) {
            if (dict.find(s) != dict.end()) res.push_back(s.substr(0,1));
            return res;
        }
        int i, j;
        for (i = 0; i < n; i++) {
            for (j = 0; j<= i; j++) {
                if (dict.find(s.substr(j, i-j+1)) != dict.end()) {
                    records[i][j] = true;
                }
            }
        }
        vector<int> delimiter;
        breakHelper(delimiter, res, s, records, n-1);
        return res;
    }
};

// Part VII: Depth First Search / NP-Problem / BackTracing
// 将问题转换成图的形式来看，对于满足要求的节点我们就选中之后进行递归。也就是深度优先搜索一条边的意思
/****** Word Search ******/
// 1. 递归的深度尽量浅，因此这种代码（一下递归好几层）就不要写，能判断条件尽量判断条件，不然肯定TLE或者runtime error
/*
    bool dfs(vector<vector<char>> &board, int i, int j, string word, int curr) {
        if (curr == word.size()) return true;
        bool out_of_edge = (i < 0) || (i >= board.size()) || (j < 0) || (j >= board[0].size());
        if (out_of_edge) return false;
        if (board[i][j] != word[curr]) return false;
        return dfs(board, i-1, j, word, curr+1) | dfs(board, i+1, j, word, curr+1) | dfs(board, i, j-1, word, curr+1) | dfs(board, i, j+1, word, curr+1);
    }
*/
// 2. 对于走不通的支路，table的点要设成unvisited，否则之后的路如果需要折回到以前的点时就会访问不了，跟循环套递归里如果走不通或者满足条件要pop_back是一个道理
class WordSearch {
public:
    bool dfs(vector<vector<char>> &board, vector<vector<bool>> &table, int i, int j, string word, int curr) {
        if (curr == word.size() - 1) return true;
        table[i][j] = false;
        if (i&&table[i-1][j]&&(word[curr+1] == board[i-1][j])) {
            if(dfs(board, table, i-1, j, word, curr+1)) return true;
            table[i-1][j] = true;
        }
        if (i+1<board.size()&&table[i+1][j]&&(word[curr+1] == board[i+1][j])) {
            if(dfs(board, table, i+1, j, word, curr+1)) return true;
            table[i+1][j] = true;
        }
        if (j&&table[i][j-1]&&(word[curr+1] == board[i][j-1])) {
            if(dfs(board, table, i, j-1, word, curr+1)) return true;
            table[i][j-1] = true;
        }
        if (j+1<board[0].size()&&table[i][j+1]&&(word[curr+1] == board[i][j+1])) {
            if(dfs(board, table, i, j+1, word, curr+1)) return true;
            table[i][j+1] = true;
        }
        return false;
    }
    
    bool exist(vector<vector<char> > &board, string word) {
        int size = word.size();
        if (size == 0) return true;
        int m = board.size();
        if (m == 0) return false;
        int n = board[0].size();
        if (n == 0) return false;
        int i, j;
        vector<vector<bool>> table(m, vector<bool>(n, true));
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                if (board[i][j] == word[0]) {
                    if(dfs(board, table, i, j, word, 0)) return true;
                    table = vector<vector<bool>>(m, vector<bool>(n, true));
                }
            }
        }
        return false;
    }
};
/*********** Find all Subsets from a given set  (the question is from CC150)**************/
// 需要注意的是最后push_back单个元素， 不要先push，否则就会重复了，例如112,33等等。
// 另外vector初始化的时候是（个数，元素值），先个数后元素，别忘
//  递归有几个要注意的点：1. base case或者end case，要清楚返回的是什么；2.递归放在哪；3.结束语句；
// ** 从这道题来看，本质上递归就是函数调用，它的目的就是求得你所需要的“值”而已，无它。其他该怎么写还怎么写。
// 想不清楚的时候还是可以先从top-down的角度考虑。例如lowest common ancestor问题。base case: 1. 没有根，2. root是p或者q。然后先考虑三个节点的二叉树：如果p,q分别为左子节点，右子节点，那祖先就是root，反之祖先就左，或者右。
class FindSubSet {
public:
	vector<vector<int>> findsubset (int A[], int n) {
		vector<vector<int>> newsubset;
		vector<vector<int>> oldsubset;
		if (n == 1) return vector<vector<int>> (1, vector<int>(1, A[0]));
		oldsubset = findsubset(A, n-1);
		newsubset = oldsubset;
		vector<vector<int>>::iterator it1;
		for (it1 = newsubset.begin(); it1!= newsubset.end(); it1++) {
			it1->push_back(A[n-1]);
		}
		for (it1 = oldsubset.begin(); it1 != oldsubset.end();it1++) {
			newsubset.push_back(*it1);
		}
		newsubset.push_back(vector<int>(1, A[n-1])); // notice the position of this instruction!!!
		return newsubset;
	}
};
/*** Combinations ***/ 
// 这个是用递归DFS的思想，当t存满k个数据后就是一个结果，把它push到result里之后返回，然后倒数第二层循环接着进一位，寻找下一个combination
class Combination {
public:
    void helper(vector<vector<int>> &res, vector<int>& t, int k, int begin, int n) {
        if (t.size() == k) {
            res.push_back(t);
            return;
        }
        for (int i = begin; i <= n; i++) {
            t.push_back(i);
            helper(res, t, k, i+1, n);
            t.pop_back();
        }
        return;
    }
    vector<vector<int> > combine(int n, int k) {
        vector<vector<int>> res;
        if (n < k) return res;
        vector<int> t;
        helper(res, t, k, 1, n);
        return res;
    }
};
/******* Solve N Queens ******/
// 这题的重点有：1。如果helper返回错了记得要把temp里的那个错的一行pop出来，不然永远是错的，output就是空集
// 2. 
class SolveNqueens {
public:
    bool helper(int currRow, int size, vector<vector<string>> &res, vector<string> &temp) {
        string curr(size, '.');
        string origin = curr;
        int i,j,k;
        bool flag = true;
        bool status = false;
        if (currRow == size) {
            res.push_back(temp);
            return true;
        }
        for (i = 0; i < size; i++) {
            if (currRow != 0) {
                for (j = currRow - 1; j >= 0; j--) {
                    for (k = 0; k < size; k++) {
						if (temp[j][k] == 'Q' && abs(currRow-j) == abs(i-k)) { flag = false; break; } // 两层循环，如果有一个地方不满足要求那就不用继续往下查了
						if (temp[j][i] == 'Q') { flag = false; break; }
                    }
					if (flag == false) break; 
                }
            }
            if (flag == true) {
                curr[i] = 'Q';
                temp.push_back(curr);
                if (!helper((currRow+1), size, res, temp)) {
					temp.pop_back();
					curr[i] = '.'; // two probs: 1. whatif there is no place to set 'Q' when traversed loop and 2. what you should do if helper returns true or false? think them carefully!
				}
                else {
					temp.pop_back();
				}
            }
            flag = true;
            curr = origin;
        }
        if (!status) return false;
        else {
            temp.pop_back();
        }
    }
    vector<vector<string> > solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> temp;
        helper(0, n, res, temp);
        return res;
    }
};
/******* Restore IP Addresses ******/
class RestoreIPAddress {
public:
    bool checkvalid(const string s, int begin, int end) {
        int num = 0;
        if (begin >= s.size()) return false;
        if (begin + 1 <= end && s[begin] == '0'){ // if there are two digits and begin is 0
            return false;
        }
        for (int i = begin; i <= end; i++) {
            num += s[i] - '0'; //这块要是再记不住 -'0'，请抽自己。
            num *= 10;
        }
        num /= 10;
        if (num >= 0 && num <= 255) return true;
        else return false;
    }
    bool helper(int start, vector<string> &res, const string s, vector<string> comp, int residual_dots) {
        int i;
        if (residual_dots == 0) {
                if (start >= s.size()) return false;
                if (!checkvalid(s, start, s.size()-1)) return false;
                string re;
                for (auto it = comp.begin(); it != comp.end(); it++) {
                    re += *it;
                }
                re += s.substr(start, s.size() - start);
                res.push_back(re);
                return true;
        }
        for (i = start; i < start + 3; i++) {
            if (i >= s.size()) return false;
            if (!checkvalid(s, start, i)) continue;
            string sub = s.substr(start, i-start+1);
            sub += '.';
            comp.push_back(sub);
            //if (!helper(i+1, res, s, comp, residual_dots-1)) comp.pop_back();
            helper(i+1, res, s, comp, residual_dots-1);
            comp.pop_back();
            //else return true;
        }
        return false;
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        vector<string> comp;
        if (s.size() == 0) return res;
        helper(0, res, s, comp, 3);
        return res;
    }
};

// Part VIII: Permutations
/*** Permutations II ***/
//算法：从尾巴开始找到最长不上升子序列，然后将此序列翻转。如果这个最长子序列就是原序列，那么翻转之后看看是不是跟原序列一样，如果跟原序列一样说明这个是"1111"这种每个元素都相同的序列，直接返回就好；如果不是原序列则继续循环；翻转好子序列之后找到子序列之前的一个元素（which implies 子序列 ！= 原序列，不然就没有元素了）。把这个元素跟翻转后的子序列的第一个比这个元素大的序列对换（ 此时注意如果整个子序列都没有比那个元素大的话，就要把这个元素放在子序列的尾巴上，这也就意味着子序列需要整体往前挪动一位（moving vector函数））。这时就成了一个新的permutation。循环的终止条件就是变换后的序列是不是跟原序列相等。如果跟原序列相等那么整个算法结束，否则继续循环。
class PermutationsII {
private:
    void swap(vector<int> &num, int p1, int p2) {
        int temp = num[p1];
        num[p1] = num[p2];
        num[p2] = temp;
    }
    
    void reverse(vector<int> &num, int p1, int p2) {
        while (p1<=p2) {
            swap(num, p1, p2);
            p1++;
            p2--;
        }
    }
    int findDesending(vector<int> &num) {
        int n = num.size();
        int i = n-1;
        while (i >0 && num[i] <= num[i-1]) i--;
        return i;
    }
    void movingVector(vector<int> &num, int dst) {
        int i;
        int dstVal = num[dst];
        for (i = dst+1; i < num.size(); i++) {
            num[i-1] = num[i];
        }
        num[num.size()-1] = dstVal;
    }
public:
    vector<vector<int> > permuteUnique(vector<int> &num) {
        vector<vector<int>> res;
        int n = num.size();
        if (n == 0) return res;
        res.push_back(num);
        if (n == 1) return res;
        vector<int> curr = num;
        vector<int> next;
        int index, tmp;
        while (1) {
            index = findDesending(curr);
            reverse(curr, index, n-1);
            if (index > 0) {
                tmp = index;
                index--;
                while (tmp < n && curr[tmp] <= curr[index]) tmp++;
                if (tmp != n) swap(curr, index, tmp);
                else movingVector(curr, index);
                if (curr != num) res.push_back(curr);
                else break;
            }
            else if (curr != num) res.push_back(curr);
            else break;
        }
        return res;
    }
};


// Part IX: Stack & Queues
/****** Evaluate Reverse Polish Notation ******/
 class EvalRPN {
public:
    int parseNum(string s) {
        int res = 0;
        int pt = 0;
        bool neg = false;
        if (s[0] == '-') { neg = true; pt++; }
        int n = s.size();
        while (pt < n) {
            int temp = s[pt] - '0';
            res += temp;
            res *= 10;
            pt++;
        }
        res/=10;
        if (neg) return 0-res;
        else return res;
    }
    
    int evalRPN(vector<string> &tokens) {
        stack<int> temp;
        if (tokens.empty()) return -1;
        if (tokens.size() == 1) return parseNum(tokens[0]);
        if (tokens.size() == 2) return -1;
        for (auto i = tokens.begin(); i != tokens.end(); i++) {
            if (*i == "+" || *i =="-" || *i == "*" || *i == "/") {
                if (temp.empty()) return -1;
                int b = temp.top();
                temp.pop();
                if (temp.empty()) return -1;
                int a = temp.top();
                temp.pop();
                int result;
                if (*i == "+") result = a+b;
                if (*i == "-") result = a-b;
                if (*i == "*") result = a*b;
                if (*i == "/") {
                    if (b != 0) result = a/b;
                    else return -1;
                }
                temp.push(result);
            }
            else {
                temp.push(parseNum(*i));
            }
        }
        int final_result = temp.top();
        temp.pop();
        if (!temp.empty()) return -1;
        else return final_result;
    }
};
// Part X: Graph
 /****** Word Ladder ******/
 // 注意一下配合哈希分层的思路
 // 真的需要锻炼不用vs debug了，对你没好处啊！
 class WordLadderI {
private:
    int findLength(string src, string dst, unordered_map<string, string> &hm) {
        int counter = 1;
        string temp;
        temp = src;
        while (hm.find(temp) != hm.end() && temp != dst) {
            temp = hm[temp];
            counter++;
        }
        if (temp == dst) return counter;
        else return 0;
    }
public:
    int ladderLength(string start, string end, unordered_set<string> &dict) {
        unordered_map<string, string> hm;
        int i, j;
        int n = start.size();
        int m = end.size();
        queue<string> level_of_words;
        assert(n == m);
        string curr, prev;
        level_of_words.push(start);
        hm[start] = "";
        while (!level_of_words.empty() && curr != end) {
            curr = level_of_words.front();
            level_of_words.pop();
            prev = curr;
            for (i = 0; i < n; i++) {
                for (j = 0; j < 26; j++) {
                    curr[i] = 'a' + j;
                    if (dict.find(curr) != dict.end() && hm.find(curr) == hm.end()) { // skip if they belongs to same level
                        level_of_words.push(curr);
                        hm[curr] = prev;
                        if (curr == end) goto END;
                    }
                }
            curr = prev;
            }
        }
        END:
        int res = findLength(end, start, hm);
        return res;
    }
};
// Part XI: Other Problems
// Counting Sort
 class SortColors {
public:
//use counting sort
void sortColors(int A[], int n) {
int red = -1, white = -1, blue = -1;

for(int i = 0; i < n; i++){
    if(A[i] == 0){   
        A[++blue] = 2;
        A[++white] = 1;
        A[++red] = 0;
    }
    else if(A[i] == 1){
        A[++blue] = 2;
        A[++white] = 1;
    }
    else if(A[i] == 2)   
        A[++blue] = 2;
}
}
};
/*********** Remove Duplicates  from Sorted Array II  **************/
/* Solution1: by zhu ge */
class RDfSA2 {
public:
    int removeDuplicates(int A[], int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if(n == 0)//别忘了这里
            return 0;
        int count = 0;
        int index = 0;//index之前是已经都弄妥了的部分
        for(int i = 1; i < n; ++i) {//这里角标从1开始
            if(A[index] != A[i]) {
                A[++index] = A[i];//记住这里
                count = 0;
            } else if(A[index] == A[i] && count == 0) {
                A[++index] = A[i];
                count++;
            } else if(A[index] == A[i] && count != 0) {
                //do nothing
            }
        }
        return index + 1;
    }
};
/* Solution 2: my awkward original solution */
class RDfSA1 {
public:
    int removeDuplicates(int A[], int n) {
        int pt1 = 0, pt2 = 1, pt3 = 2;
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;
        if (n == 3) {
            if (A[pt1] == A[pt2] && A[pt1] == A[pt3]) return 2;
            else return 3;
        }
        int counter = 0;
        for (pt1 = 0, pt2 = 1, pt3 = 2; pt3 < n; pt1++, pt2++, pt3++) {
            if (A[pt1] == A[pt2] && A[pt1] == A[pt3]) {
                counter++;
                while (A[pt1+counter] == A[pt2+counter] && A[pt1+counter] == A[pt3+counter] && pt3 + counter < n) counter++;
                if (pt3 < n - counter) {
                    for (int i = pt3 + counter; i < n; i++) {
                        A[i-counter] = A[i];
                    }
                    n -= counter;
                    counter = 0;
                }
                else if (pt3 == n-counter) break;
            }
        }
        return n - counter;
    }
};
/******* Reverse Words in a String *******/
 class ReverseWords {
public:
    void reverseWords(string &s)
{
    string rs;
    for (int i = s.length()-1; i >= 0; )
    {
        while (i >= 0 && s[i] == ' ') i--;
        if (i < 0) break;
        if (!rs.empty()) rs.push_back(' ');
        string t;
        while (i >= 0 && s[i] != ' ') t.push_back(s[i--]);
        reverse(t.begin(), t.end());
        rs.append(t);
    }
    s = rs;
}
};
 /*********** Pascal's Triangle ************/
class PT1 {
public:
    vector<vector<int> > generate(int numRows) {
        vector<vector<int>> res;
        int i, j;
        if (numRows <= 0) return res;
        
        for (i = 1; i <= 2; i++) {
            vector<int> tmp(i, 1);
            res.push_back(tmp);
            if (numRows == 1) return res;
        }
        if (numRows == 2) return res;
        for (i = 2; i < numRows; i++) {
            vector<int> tmp(i+1, 1);
            vector<int> prev = res[i-1];
            for (j = 1; j < i; j++) {
                tmp[j] = prev[j] + prev[j-1];
            }
            res.push_back(tmp);
        }
        return res;
    }
}; 
 /********** Plus One ***********/
class PlusOne {
public:
    vector<int> plusOne(vector<int> &digits) {
        vector<int> res;
        if (digits.size() == 0) return res;
        int n = digits.size();
        int sum = 1;
        for (int i = n-1; i>=0; i--) {
            if (digits[i]+sum == 10) {
                digits[i] = 0;
                sum = 1;
            }
            else {
                digits[i] += sum;
                sum = 0;
            }
        }
        if (sum == 1) digits.insert(digits.begin(), 1);
        res = digits;
        return res;
    }
};
 /********** Regular Expression Matching ***********/
// 这个题我觉得很奇怪，OJ上说"ab", ".*c"这个不算match，但明明它其实是match的不是么。。。
// OJ的给的几个例子：（都测试通过了）
/*
Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "a*") → true
isMatch("aa", ".*") → true
isMatch("ab", ".*") → true
isMatch("aab", "c*a*b") → true
*/
// 看倒数第二个例子，OJ给的true, 那"ab", ".*c"不明显也是true么？不明白
// 可能明白它的意思了。。。需要把P上的字符串全部cover掉，不能有遗漏
class RegExpMatching {
public:
    bool isMatch(const char *s, const char *p) {
        string s1(s);
		string s2(p);
		int m = s1.size();
		int n = s2.size();
		// vector<vector<bool>> records(m, vector<bool>(n, false));
		// so the idea is we return sum (j = 0~n-1) records[m-1][j], whatever the j's position, as long as it is true, it means the s is covered by p 
		// so how to proceed it ?
		// suppose [i-1] is matched, then [i] is matched iff: 
		// s1[i] == s2[j] || (s2[j-1] = ='.' || s2[j-1] ==s1[i]) && s2[j] == '*' || s2[j] == '.'
		// notice the status of '*', if is not matched, do not clear the pointer i
		// only '*' should be expended forward to more chars, otherwise matched one by one 
		int i = 0, j = 0;
		// bool processing = false; // to determine whether s is matching...
		// processing
		bool star_pattern = false;
		while (i < m && j < n) {
			if (s1[i] == s2[j] || s2[j] == '.') {
				i++;
				j++;
				continue;
			}
			while ( j>0 && i < m && ((s2[j-1] == '.' || s2[j-1] == s1[i]) && s2[j] == '*')) {
				star_pattern= true;
				i++;
			}
			if (star_pattern) {
				if (i >= m) break;
				else {
					i++;
					j++;
					star_pattern = false;
					continue;
				}
			}
			if ( s2[j] == '*' && (j ==0 || (j > 0 && s2[j-1] != '.' && s2[j-1] != s1[i]))) {
				j++;
				continue;
			}
			i = 0;
			j++;
		}
		return (i >= m)?true:false;
    }
};
int main () {
	return 0;
} 