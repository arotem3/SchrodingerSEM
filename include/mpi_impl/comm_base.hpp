#ifndef MPI_IMPL_COMM_BASE_HPP
#define MPI_IMPL_COMM_BASE_HPP

#include <concepts>
#include <set>
#include <unordered_map>
#include <vector>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include "types.hpp"
#include "CornerNode.hpp"

namespace schro_mpi
{
    namespace mpi = boost::mpi;

    template <std::floating_point real>
    struct EdgeMessage
    {
        friend class boost::serialization::access;

        int id;
        std::vector<real> values;

        EdgeMessage() {}
        EdgeMessage(int e, const std::vector<real>& v) : id(e), values(v) {}
        EdgeMessage(int e, std::vector<real>&& v) : id(e), values(std::move(v)) {}

        template<class Archive>
        void serialize(Archive& ar, const u_int v)
        {
            ar & id;
            ar & values;
        }
    };

    template <std::floating_point real>
    struct CornerMessage
    {
        real value;
        ElementCornerInfo info;

        CornerMessage() {}
        CornerMessage(real val, const ElementCornerInfo& inf) : value(val), info(inf) {}

        template<class Archive>
        void serialize(Archive& ar, const u_int v)
        {
            ar & value;
            ar & info;
        }
    };

    template <std::floating_point real>
    using EdgePromises = std::unordered_map<int, std::vector<EdgeMessage<real>>>;
    
    template <std::floating_point real>
    using CornerPromises = std::unordered_map<int, std::vector<CornerMessage<real>>>;

    template <std::floating_point real>
    using CornerSumPromises = std::unordered_map<int, std::vector<std::pair<int, real>>>;

} // namespace schro_mpi

namespace boost
{
    namespace serialization
    {
        template <class Archive>
        void serialize(Archive& ar, ElementCornerInfo& info, const unsigned int version)
        {
            ar & info.element_id;
            ar & info.i;
            ar & info.j;
        }

        template <std::floating_point real, class Archive>
        void serialize(Archive& ar, CornerNode<real>& node, const unsigned int version)
        {
            ar & node.corner_type;
            ar & node.x;
            ar & node.y;
            ar & node.connected_elements;
        }

        template <class Archive>
        void serialize(Archive& ar, Edge& edge, const unsigned int version)
        {
            ar & edge.edge_type;
            ar & edge.elements;
            ar & edge.element_sides;
            ar & edge.startIter;
            ar & edge.deltaIter;
        }

        template <std::floating_point real, class Archive>
        void serialize(Archive& ar, Quad<real>& element, const unsigned int version)
        {
            ar & element.xs;
            ar & element.ys;
            ar & element.corner_ids;
            ar & element.n;
        }
        
    } // namespace serialization
} // namespace boost

#endif