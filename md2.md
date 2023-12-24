# Corteza JS API Documentation

All of Corteza's functions are exposed via three api’s (System, Compose, Federation).
The decomposition of these api’s can be found at https://cortezaurl/api/docs/#/ and we can choose between them by the menu on the top right:

![Screenshot](assets/screenshot1.png)

and below we can find the details of each api :

* System:
  * Authentication
  * Authentication clients
  * Settings
  * Subscription
    ...
* Compose:
  * Namespaces
  * Pages
  * Modules
  * Records
    ...
* Federation:
  * Federation node handshake
  * Federation nodes
  * Manage structure
  * Sync structure
    ...

to call this apis from js code we can use some utils modules defined in cortezajs lib
The complete definition of this modules can be found in the following path in the corteza repo :
corteza/lib/js/src/api-clients/

* automation.ts
* compose.ts
* federation.ts
* system.ts

For example, in the compose module, the definition of namespaceList function is the following:

```javascript
async namespaceList (a: KV, extra: AxiosRequestConfig = {}): Promise<KV> {
const { query, slug, limit,incTotal,labels,pageCursor,sort,} = (a as KV) || {}
const cfg: AxiosRequestConfig = {...extra,method: 'get',url:this.namespaceListEndpoint(),}
cfg.params = {query,slug,limit,incTotal,labels,pageCursor,sort,}
return this.api().request(cfg).then(result => stdResolve(result))
}
```

In the context of corteza compose components, these module are expose using a wrapper objects.

* `this.$ComposeAPI` for Compose api
* `this.$SystemAPI` for System API
* `this.$FederationAPI` for FederationAPI

Exemples

namespaceList

search for a Namespace

```javascript
this.$ComposeAPI.namespaceList({ slug: "crm" })
.then(({ set = [] }) => {
const [ns] = set;
if (ns) {
console.log(ns);
}
})
.catch(() => {
this.toastErrorHandler(“error message"));

});
```

All namepces

```javascript
this.$ComposeAPI.namespaceList()
.then(({ set = [] }) => {
const [ns] = set;
if (ns) {
console.log(ns);
}
})
.catch(() => {
this.toastErrorHandler(“error message"));
});
```

recordList
to Read records of a module we can use the recordList function with the following parameters:

- namespaceID:namespace id
- moduleID: module id
- query  : String that respect the Query Language https://docs.cortezaproject.org/corteza-docs/2022.9/integrator-guide/accessing-corteza/ql.html
- deleted: Exclude (0, default), include (1) or return only (2) deleted records
- incTotal: boolean ton include total count
- incPageNavigation:include page navigation
- limite: results limits   
- pageCursor:page Cursor
- sort : Sort results (we can add DESC for descending sorting)

```javascript
await await this.$ComposeAPI
  .recordList({moduleID, namespaceID, query: "name like '%aname'",incTotal: true,incPageNavigation: true,limit: 100, sort: "name"
  })
  .then(({ set, filter }) => {
    console.log(set);
})
.catch(
this.toastErrorHandler(this.$t("notification:record.listLoadFailed"))
)
.finally(() => {
this.processing = false;
});
```
